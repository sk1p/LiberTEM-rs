use std::{
    collections::HashMap,
    iter,
    marker::PhantomData,
    mem::size_of,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, RwLock},
};

use bytes::BytesMut;
use log::debug;
use ndarray::{ArrayViewMut, Axis, Slice};
use num::cast::AsPrimitive;
use serde::{Deserialize, Serialize};
use zarrs::{
    array::{
        codec::{
            bytes_to_bytes::blosc::{BloscCompressionLevel, BloscCompressor},
            BloscCodec, ZstdCodec,
        },
        ArrayBuilder, DataType, FillValue,
    },
    storage::{
        store::{FilesystemStore, FilesystemStoreOptions},
        ReadableWritableListableStorage, ReadableWritableListableStorageTraits, StoreKey,
    },
};
use zerocopy::AsBytes;

use crate::{
    consumer::{Consumer, ConsumerError},
    decoder::{Decoder, DecoderTargetPixelType},
    frame_stack::{FrameMeta, FrameStackHandle},
};

pub struct ZarrWriter {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ZarrCompression {
    Blosc {
        cname: BloscCompressor,
        clevel: BloscCompressionLevel,
    },
    Zstd {
        level: i32,
        checksum: bool,
    },
}

impl ZarrWriter {
    pub fn init(
        save_path: &Path,
        array_path: &str,
        compression: Option<&ZarrCompression>,
        chunk_grid: &[u64],
        array_shape: &[u64],
        dtype: DataType,
        fill_value: FillValue,
    ) -> Result<(), ConsumerError> {
        let mut opts = FilesystemStoreOptions::default();
        opts.direct_io(true);
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new_with_options(save_path, opts).unwrap());

        let chunk_grid = chunk_grid.to_vec();
        let shape = array_shape.to_vec();

        let mut builder =
            ArrayBuilder::new(shape, dtype, chunk_grid.try_into().unwrap(), fill_value);

        let array = match compression {
            None => builder.build(Arc::clone(&store), array_path)?,
            Some(ZarrCompression::Zstd { level, checksum }) => {
                let array = builder
                    .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(*level, *checksum))])
                    .build(Arc::clone(&store), array_path)?;
                array
            }
            Some(ZarrCompression::Blosc { cname, clevel }) => {
                let array = builder
                    .bytes_to_bytes_codecs(vec![Arc::new(
                        BloscCodec::new(
                            *cname,
                            *clevel,
                            None,
                            zarrs::array::codec::bytes_to_bytes::blosc::BloscShuffleMode::NoShuffle,
                            None,
                        )
                        .unwrap(),
                    )])
                    .build(Arc::clone(&store), array_path)?;
                array
            }
        };

        array.store_metadata()?;
        debug!("array metadata stored");
        Ok(())
    }
}

/// A `Consumer` that buffers the data and writes it as a single chunk into a
/// zarr array.
pub struct ZarrChunkWriter<T, D, M>
where
    T: DecoderTargetPixelType,
    u8: AsPrimitive<T>,
    u16: AsPrimitive<T>,
    D: Decoder + Send + Sync,
    M: FrameMeta + Send + Sync,
{
    arr: zarrs::array::Array<dyn ReadableWritableListableStorageTraits>,
    store: ReadableWritableListableStorage,
    save_path: PathBuf,
    files: Mutex<HashMap<StoreKey, Arc<RwLock<()>>>>,
    chunk_buf: Option<BytesMut>,
    _d: PhantomData<D>,
    _m: PhantomData<M>,
    _t: PhantomData<T>,
}

impl<T, D, M> ZarrChunkWriter<T, D, M>
where
    T: DecoderTargetPixelType,
    u8: AsPrimitive<T>,
    u16: AsPrimitive<T>,
    D: Decoder + Send + Sync,
    M: FrameMeta + Send + Sync,
{
    pub fn new(save_path: &Path, array_path: &str) -> Self {
        let mut opts = FilesystemStoreOptions::default();
        opts.direct_io(true);
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new_with_options(save_path, opts).unwrap());
        let arr = zarrs::array::Array::open(Arc::clone(&store), array_path).unwrap();
        Self {
            arr,
            store,
            save_path: save_path.to_owned(),
            files: Mutex::new(HashMap::new()),
            chunk_buf: None,
            _d: Default::default(),
            _m: Default::default(),
            _t: Default::default(),
        }
    }
}

fn bytes_aligned(size: usize) -> BytesMut {
    let align = page_size::get();
    let mut bytes = BytesMut::with_capacity(size + 2 * align);
    let offset = bytes.as_ptr().align_offset(align);
    bytes.split_off(offset)
}

impl<T, D, M> Consumer for ZarrChunkWriter<T, D, M>
where
    T: DecoderTargetPixelType + Send + Sync + zarrs::array::Element,
    u8: AsPrimitive<T>,
    u16: AsPrimitive<T>,
    D: Decoder<FrameMeta = M> + Send + Sync,
    M: FrameMeta + Send + Sync,
{
    type Decoder = D;
    type FrameMeta = M;

    fn consume_frame_stack(
        &mut self,
        handle: &FrameStackHandle<M>,
        decoder: &D,
        shm: &ipc_test::SharedSlabAllocator,
    ) -> Result<(), ConsumerError> {
        let len: usize = handle.len();
        let shape = handle.first_meta().get_shape();

        // FIXME: if len != chunk size, we need to buffer
        assert_eq!(len, 16);

        let buf = self.chunk_buf.take();
        let size_bytes = len * (shape.0 * shape.1) as usize * size_of::<T>();

        let mut buf = if let Some(buf) = buf {
            if buf.len() < size_bytes {
                let mut buf = bytes_aligned(size_bytes);
                buf.extend(iter::repeat(0).take(size_bytes));
                buf
            } else {
                buf
            }
        } else {
            let mut buf = bytes_aligned(size_bytes);
            buf.extend(iter::repeat(0).take(size_bytes));
            buf
        };

        let buf_t = T::mut_slice_from(buf.as_bytes_mut()).unwrap();

        let mut view =
            ArrayViewMut::from_shape([len, shape.0 as usize, shape.1 as usize], buf_t).unwrap();
        let mut output = view.slice_axis_mut(Axis(0), Slice::from(0..len));
        decoder.decode(shm, handle, &mut output, 0, len).unwrap();
        let chunk_indices = [(handle.first_meta().get_index() / len) as u64, 0u64, 0u64];

        let buf_frozen = buf.freeze();
        unsafe {
            self.arr
                .store_encoded_chunk(&chunk_indices, buf_frozen.clone())
                .unwrap();
        }
        buf = buf_frozen
            .try_into_mut()
            .expect("we are the only user of this buffer");

        self.chunk_buf = Some(buf);

        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ConsumerError> {
        Ok(())
    }
}

impl<T, D, M> ZarrChunkWriter<T, D, M>
where
    T: DecoderTargetPixelType + Send + Sync + zarrs::array::Element,
    u8: AsPrimitive<T>,
    u16: AsPrimitive<T>,
    D: Decoder<FrameMeta = M> + Send + Sync,
    M: FrameMeta + Send + Sync,
{
}
