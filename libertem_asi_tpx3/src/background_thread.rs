use std::{
    fmt::Display,
    mem::replace,
    net::TcpStream,
    sync::mpsc::{channel, Receiver, RecvTimeoutError, SendError, Sender, TryRecvError},
    thread::JoinHandle,
    time::{Duration, Instant},
};

use common::{
    background_thread::{BackgroundThread, ControlMsg, ReceiverMsg},
    generic_connection::{AcquisitionConfig, GenericConnection},
};
use ipc_test::{SHMHandle, SharedSlabAllocator};
use log::{debug, error, info, trace, warn};

use crate::{
    chunk_stack::{ChunkCSRLayout, ChunkStackForWriting, ChunkStackHandle, TPXFrameMeta},
    csr_view_raw::CSRViewRaw,
    headers::{AcquisitionEnd, AcquisitionStart, HeaderTypes, ScanEnd, ScanStart},
    stream::{stream_recv_chunk, stream_recv_header, StreamError},
};

pub type TPXControlMsg = ControlMsg<()>;
pub type TPXReceiverMsg = ReceiverMsg<TPXFrameMeta, AcquisitionStart>;

#[derive(PartialEq, Eq)]
pub enum ReceiverStatus {
    Idle,
    Running,
    Closed,
}

#[derive(thiserror::Error, Debug)]
enum AcquisitionError {
    Disconnected,
    Cancelled,
    BufferFull,
    SlotSizeTooSmall {
        slot_size: usize,
        chunk_size: usize,
    },

    /// Example: an unexpected header was received
    StateError {
        msg: String,
    },

    /// An error occurred while reading the socket
    StreamError {
        err: StreamError,
    },
}

impl From<StreamError> for AcquisitionError {
    fn from(err: StreamError) -> Self {
        AcquisitionError::StreamError { err }
    }
}

impl<T> From<SendError<T>> for AcquisitionError {
    fn from(_value: SendError<T>) -> Self {
        AcquisitionError::Disconnected
    }
}

impl Display for AcquisitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcquisitionError::Cancelled => {
                write!(f, "acquisition cancelled")
            }
            AcquisitionError::SlotSizeTooSmall {
                slot_size,
                chunk_size,
            } => {
                write!(f, "slot size {slot_size} too small for chunk {chunk_size}")
            }
            AcquisitionError::Disconnected => {
                write!(f, "other end has disconnected")
            }
            AcquisitionError::BufferFull => {
                write!(f, "shm buffer is full")
            }
            AcquisitionError::StateError { msg } => {
                write!(f, "state error: {msg}")
            }
            AcquisitionError::StreamError { err } => {
                write!(f, "stream error: {err:?}")
            }
        }
    }
}

/// With a running acquisition, check for control messages;
/// especially convert `ControlMsg::StopThread` to `AcquisitionError::Cancelled`.
fn check_for_control(control_channel: &Receiver<TPXControlMsg>) -> Result<(), AcquisitionError> {
    match control_channel.try_recv() {
        Ok(TPXControlMsg::StartAcquisitionPassive) => Err(AcquisitionError::StateError {
            msg: "received StartAcquisitionPassive while an acquisition was already running"
                .to_owned(),
        }),
        Ok(TPXControlMsg::StopThread) => Err(AcquisitionError::Cancelled),
        Ok(TPXControlMsg::SpecializedControlMsg { msg }) => Err(AcquisitionError::StateError {
            msg: format!("received unknowen SpecializedControlMsg: {msg:?}").to_owned(),
        }),
        Err(TryRecvError::Disconnected) => Err(AcquisitionError::Cancelled),
        Err(TryRecvError::Empty) => Ok(()),
    }
}

/// Passively listen for global acquisition and scan headers
/// and automatically latch on to them.
fn wait_for_acquisition(
    control_channel: &Receiver<TPXControlMsg>,
    from_thread_s: &Sender<TPXReceiverMsg>,
    stream: &mut TcpStream,
    frame_stack_size: usize,
    shm: &mut SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    // message headers are always 32 bytes; so we put them on the stack here:
    let mut header_bytes: [u8; 32] = [0; 32];
    loop {
        info!("waiting for acquisition header...");
        let header = stream_recv_header(stream, &mut header_bytes, || {
            check_for_control(control_channel)
        })?;

        trace!("got a header: {header:?}");

        match header {
            HeaderTypes::AcquisitionStart { header } => {
                match wait_for_scan(
                    header,
                    control_channel,
                    from_thread_s,
                    stream,
                    frame_stack_size,
                    shm,
                ) {
                    Ok(_) => {}
                    // need to bubble up stream errors without sending them to the main thread:
                    Err(e @ AcquisitionError::StreamError { .. }) => {
                        return Err(e);
                    }
                    Err(e) => {
                        let msg = format!("Error while an acquisition was running: {e:?}");
                        from_thread_s
                            .send(TPXReceiverMsg::FatalError { error: Box::new(e) })
                            .unwrap();
                        return Err(e);
                    }
                }

                let free = shm.num_slots_free();
                let total = shm.num_slots_total();
                info!("passive acquisition done; free slots: {}/{}", free, total);
            }
            other => {
                // we haven't received an `AcquisitionStart` header, so we don't
                // know the size of array chunks => we can't synchronize to the stream
                // again, so we need to bail out:
                let msg = format!("expected `AcquisitionStart` header, got {other:?}");
                return Err(AcquisitionError::StateError { msg });
            }
        }

        check_for_control(control_channel)?;
    }
}

fn wait_for_scan(
    acquisition_header: AcquisitionStart,
    control_channel: &Receiver<TPXControlMsg>,
    from_thread_s: &Sender<TPXReceiverMsg>,
    stream: &mut TcpStream,
    frame_stack_size: usize,
    shm: &mut SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    // message headers are always 32 bytes; so we put them on the stack here:
    let mut header_bytes: [u8; 32] = [0; 32];
    info!("waiting for scan header...");
    loop {
        let header = stream_recv_header(stream, &mut header_bytes, || {
            check_for_control(control_channel)
        })?;
        trace!("got a header: {header:?}");

        match header {
            HeaderTypes::ScanStart { header } => {
                match handle_scan(
                    acquisition_header.clone(),
                    header,
                    control_channel,
                    from_thread_s,
                    stream,
                    frame_stack_size,
                    shm,
                ) {
                    Ok(ScanResult::ScanCancelled(header)) => {
                        info!(
                            "AcquisitionEnd header received for sequence {}",
                            header.sequence
                        );
                        return Ok(());
                    }
                    Ok(ScanResult::ScanDone(_header)) => {}
                    Err(e) => return Err(e),
                };

                let free = shm.num_slots_free();
                let total = shm.num_slots_total();
                info!("passive scan done; free slots: {}/{}", free, total);
            }
            HeaderTypes::AcquisitionEnd { header } => {
                info!(
                    "AcquisitionEnd header received for sequence {}",
                    header.sequence
                );
                return Ok(());
            }
            other => {
                // for now, be strict about what headers we expect:
                let msg = format!("expected `ScanStart` header, got {other:?}");
                return Err(AcquisitionError::StateError { msg });
            }
        }

        check_for_control(control_channel)?;
    }
}

enum ScanResult {
    ScanDone(ScanEnd),
    ScanCancelled(AcquisitionEnd),
}

///
/// Handle one scan: the `AcquisitionStart` and `ScanStart` messages should
/// already be consumed, the following messages should be:
///
/// - one or more `ArrayChunk` messages and their payload data
/// - exactly one `ScanEnd`
///
/// We don't consume any more messages; the caller should then handle either
///
/// - `AcquisitionEnd` if this was the last scan of the acquisition
/// - `ScanStart` if another scan follows
///
fn handle_scan(
    acquisition_header: AcquisitionStart,
    _scan_header: ScanStart,
    to_thread_r: &Receiver<TPXControlMsg>,
    from_thread_s: &Sender<TPXReceiverMsg>,
    stream: &mut TcpStream,
    chunks_per_stack: usize,
    shm: &mut SharedSlabAllocator,
) -> Result<ScanResult, AcquisitionError> {
    let t0 = Instant::now();
    let mut last_control_check = Instant::now();

    trace!("sending TPXReceiverMsg::AcquisitionStart message");
    from_thread_s.send(TPXReceiverMsg::AcquisitionStart {
        pending_acquisition: acquisition_header.clone(),
    })?;

    let slot = match shm.get_mut() {
        None => return Err(AcquisitionError::BufferFull),
        Some(x) => x,
    };
    let mut chunk_stack = ChunkStackForWriting::new(slot, chunks_per_stack);

    let mut frame_counter: u64 = 0;

    loop {
        if last_control_check.elapsed() > Duration::from_millis(100) {
            last_control_check = Instant::now();
            check_for_control(to_thread_r)?;
        }

        let mut header_bytes: [u8; 32] = [0; 32];
        let header =
            stream_recv_header(stream, &mut header_bytes, || check_for_control(to_thread_r))?;

        match header {
            HeaderTypes::ArrayChunk { header } => {
                let nbytes = header.get_chunk_size_bytes(&acquisition_header);

                if chunk_stack.total_size() < nbytes {
                    // chunk is too large for the slot size, bail out...
                    return Err(AcquisitionError::SlotSizeTooSmall {
                        slot_size: chunk_stack.total_size(),
                        chunk_size: nbytes,
                    });
                }

                if !chunk_stack.can_fit(nbytes) {
                    // send to our queue:
                    let handle = {
                        let slot = match shm.get_mut() {
                            None => return Err(AcquisitionError::BufferFull),
                            Some(x) => x,
                        };
                        let new_frame_stack = ChunkStackForWriting::new(slot, chunks_per_stack);
                        let old_chunk_stack = replace(&mut chunk_stack, new_frame_stack);
                        old_chunk_stack.writing_done(shm)
                    };

                    // XXX debugging stuff here....
                    // {
                    //     let slot_r = shm.get(handle.slot.slot_idx);
                    //     assert_eq!(
                    //         handle.total_bytes_used - handle.total_bytes_padding,
                    //         handle.get_layout().iter().map(|l| l.data_length_bytes).sum()
                    //     );
                    //     handle.get_chunk_views_raw(&slot_r);
                    // }

                    from_thread_s.send(TPXReceiverMsg::FrameStack {
                        frame_stack: handle,
                    })?;
                }

                trace!("got an ArrayChunk of size {nbytes} bytes at {frame_counter}; size={}; header={header:?}", header.nframes);
                frame_counter += header.nframes as u64;

                let sizes = header.get_sizes(&acquisition_header);
                let layout = ChunkCSRLayout {
                    indptr_dtype: acquisition_header.indptr_dtype,
                    indptr_offset: 0,
                    indptr_size: sizes.indptr,
                    indices_dtype: acquisition_header.indices_dtype,
                    indices_offset: header.indices_offset as usize,
                    indices_size: sizes.indices,
                    value_dtype: header.value_dtype,
                    value_offset: header.values_offset as usize,
                    value_size: sizes.values,
                    nframes: header.nframes,
                    nnz: header.length,
                    data_length_bytes: nbytes,
                };
                let buf = chunk_stack.slice_for_writing(nbytes, layout.clone());
                stream_recv_chunk(stream, buf, || check_for_control(to_thread_r))?;
                CSRViewRaw::from_bytes_with_layout(buf, &layout);
            }
            HeaderTypes::ScanEnd { header } => {
                let elapsed = t0.elapsed();
                info!("scan done in {elapsed:?}",);

                let handle = chunk_stack.writing_done(shm);
                from_thread_s.send(TPXReceiverMsg::End {
                    frame_stack: handle,
                })?;
                return Ok(ScanResult::ScanDone(header));
            }
            HeaderTypes::AcquisitionEnd { header } => {
                let elapsed = t0.elapsed();
                warn!("AcquisitionEnd after {elapsed:?} - probably cancelled?",);
                let handle = chunk_stack.writing_done(shm);
                // FIXME: send a different result message in this case?
                from_thread_s.send(TPXReceiverMsg::Finished {
                    frame_stack: handle,
                })?;
                return Ok(ScanResult::ScanCancelled(header));
            }
            other_header => {
                let msg = format!("unexpected header in acquisition: {other_header:?}");
                return Err(AcquisitionError::StateError { msg });
            }
        }
    }
}

/// convert `AcquisitionError`s to messages on `from_threads_s`
fn background_thread_wrap(
    to_thread_r: &Receiver<TPXControlMsg>,
    from_thread_s: &Sender<TPXReceiverMsg>,
    uri: &str,
    frame_stack_size: usize,
    shm: SharedSlabAllocator,
) {
    if let Err(err) = background_thread(to_thread_r, from_thread_s, uri, frame_stack_size, shm) {
        log::error!("background_thread err'd: {}", err.to_string());
        // NOTE: `shm` is dropped in case of an error, so anyone who tries to connect afterwards
        // will get an error
        from_thread_s
            .send(TPXReceiverMsg::FatalError {
                error: Box::new(err),
            })
            .unwrap();
    }
}

/// Make a stream, returning None on timeout
fn make_stream(remote: &str) -> Option<TcpStream> {
    info!("connecting to {remote}");
    let stream = match TcpStream::connect(remote) {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to connect to {remote} ({e})");
            return None;
        }
    };
    info!("Connected to {remote}");
    stream
        .set_read_timeout(Some(Duration::from_millis(100)))
        .unwrap();
    Some(stream)
}

fn passive_loop(
    to_thread_r: &Receiver<TPXControlMsg>,
    from_thread_s: &Sender<TPXReceiverMsg>,
    remote: &str,
    frame_stack_size: usize,
    shm: &mut SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    loop {
        let mut stream = loop {
            match make_stream(remote) {
                Some(s) => break s,
                None => {
                    check_for_control(to_thread_r)?;
                    std::thread::sleep(Duration::from_secs(1));
                    continue;
                }
            }
        };

        'inner: loop {
            match wait_for_acquisition(
                to_thread_r,
                from_thread_s,
                &mut stream,
                frame_stack_size,
                shm,
            ) {
                Ok(_) => {}
                Err(AcquisitionError::Disconnected) => {
                    return Ok(()); // the other end of the channel is gone, so :shrug:
                }
                Err(AcquisitionError::StreamError { err }) => {
                    // special case: if the acquisition has been cancelled,
                    // don't try to reconnect
                    if let StreamError::ControlError(ControlError::Cancelled) = err {
                        return Err(AcquisitionError::Cancelled);
                    }
                    // FIXME: should probably bubble this one up further, as it
                    // may leave buffer contents "undefined"... => clean SHM is best
                    error!("Got a stream error: {err:?} - reconnecting...");
                    break 'inner;
                }
                e => {
                    return e; // other error, give up? maybe we can retry instead?
                }
            }
        }
    }
}

fn background_thread(
    to_thread_r: &Receiver<TPXControlMsg>,
    from_thread_s: &Sender<TPXReceiverMsg>,
    remote: &str,
    frame_stack_size: usize,
    mut shm: SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    loop {
        // control: main threads tells us to start or quit
        let control = to_thread_r.recv_timeout(Duration::from_millis(100));
        match control {
            Ok(TPXControlMsg::StartAcquisitionPassive) => {
                match passive_loop(
                    to_thread_r,
                    from_thread_s,
                    remote,
                    frame_stack_size,
                    &mut shm,
                ) {
                    Ok(_) => {}
                    Err(AcquisitionError::Cancelled) => {
                        info!("Cancelled - stopping thread and closing connection...");
                        break;
                    }
                    e @ Err(_) => return e,
                }
            }
            Ok(TPXControlMsg::StopThread) => {
                debug!("background_thread: got a StopThread message");
                break;
            }
            Ok(TPXControlMsg::SpecializedControlMsg { msg }) => {
                error!("unexpected SpecializedControlMsg: {msg:?}");
                break;
            }
            Err(RecvTimeoutError::Disconnected) => {
                debug!("background_thread: control channel has disconnected");
                break;
            }
            Err(RecvTimeoutError::Timeout) => (), // no message, nothing to do
        }
    }
    debug!("background_thread: is done");
    Ok(())
}

pub struct ReceiverError {
    pub msg: String,
}

impl Display for ReceiverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = &self.msg;
        write!(f, "{msg}")
    }
}

/// Start a background thread that received data from the zeromq socket and
/// puts it into shared memory.
pub struct TPXReceiver {
    bg_thread: Option<JoinHandle<()>>,
    to_thread: Sender<TPXControlMsg>,
    from_thread: Receiver<TPXReceiverMsg>,
    pub status: ReceiverStatus,
    pub shm_handle: SHMHandle,
}

impl TPXReceiver {
    pub fn new(uri: &str, frame_stack_size: usize, shm: SharedSlabAllocator) -> Self {
        let (to_thread_s, to_thread_r) = channel();
        let (from_thread_s, from_thread_r) = channel();

        let builder = std::thread::Builder::new();
        let uri = uri.to_string();

        let shm_handle = shm.get_handle();

        TPXReceiver {
            bg_thread: Some(
                builder
                    .name("bg_thread".to_string())
                    .spawn(move || {
                        background_thread_wrap(
                            &to_thread_r,
                            &from_thread_s,
                            &uri,
                            frame_stack_size,
                            shm,
                        )
                    })
                    .expect("failed to start background thread"),
            ),
            from_thread: from_thread_r,
            to_thread: to_thread_s,
            status: ReceiverStatus::Idle,
            shm_handle,
        }
    }

    fn adjust_status(&mut self, msg: &TPXReceiverMsg) {
        match msg {
            TPXReceiverMsg::AcquisitionStart { .. } => {
                self.status = ReceiverStatus::Running;
            }
            TPXReceiverMsg::Finished { .. } => {
                self.status = ReceiverStatus::Idle;
            }
            _ => {}
        }
    }

    pub fn recv(&mut self) -> TPXReceiverMsg {
        let result_msg = self
            .from_thread
            .recv()
            .expect("background thread should be running");
        self.adjust_status(&result_msg);
        result_msg
    }

    pub fn next_timeout(&mut self, timeout: Duration) -> Option<TPXReceiverMsg> {
        let result_msg = self.from_thread.recv_timeout(timeout);

        match result_msg {
            Ok(result) => {
                self.adjust_status(&result);
                Some(result)
            }
            Err(e) => match e {
                RecvTimeoutError::Disconnected => {
                    panic!("background thread should be running")
                }
                RecvTimeoutError::Timeout => None,
            },
        }
    }

    pub fn start_passive(&mut self) -> Result<(), ReceiverError> {
        if self.status == ReceiverStatus::Closed {
            return Err(ReceiverError {
                msg: "receiver is closed".to_string(),
            });
        }
        self.to_thread
            .send(TPXControlMsg::StartAcquisitionPassive)
            .expect("background thread should be running");
        self.status = ReceiverStatus::Running;
        Ok(())
    }

    pub fn close(&mut self) {
        if self.to_thread.send(TPXControlMsg::StopThread).is_err() {
            warn!("could not stop background thread, probably already dead");
        }
        if let Some(join_handle) = self.bg_thread.take() {
            join_handle
                .join()
                .expect("could not join background thread!");
        } else {
            warn!("did not have a bg thread join handle, cannot join!");
        }
        self.status = ReceiverStatus::Closed;
    }
}

struct TXPBackgroundThread {
    bg_thread: JoinHandle<()>,
    to_thread: Sender<TPXControlMsg>,
    from_thread: Receiver<TPXReceiverMsg>,
    pub status: ReceiverStatus,
    pub shm_handle: SHMHandle,
}

impl BackgroundThread for TXPBackgroundThread {
    type FrameMetaImpl = TPXFrameMeta;

    type AcquisitionConfigImpl = AcquisitionStart;

    type ExtraControl = ();

    fn channel_to_thread(
        &mut self,
    ) -> &mut std::sync::mpsc::Sender<common::background_thread::ControlMsg<Self::ExtraControl>>
    {
        &mut self.to_thread
    }

    fn channel_from_thread(
        &mut self,
    ) -> &mut std::sync::mpsc::Receiver<
        common::background_thread::ReceiverMsg<Self::FrameMetaImpl, Self::AcquisitionConfigImpl>,
    > {
        &mut self.from_thread
    }

    fn join(self) {
        if let Err(e) = self.bg_thread.join() {
            // FIXME: should we have an error boundary here instead and stop the panic?
            std::panic::resume_unwind(e)
        }
    }
}

pub struct TPXConnection {
    _inner: GenericConnection<TXPBackgroundThread, AcquisitionStart>,
}

impl TPXConnection {
    fn new() -> Self {
        Self {
            _inner: GenericConnection::new(bg_thread, shm),
        }
    }
}
