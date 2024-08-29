use std::{
    cell::RefCell,
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
use zarrs::{
    array::{
        codec::{array_to_bytes::sharding::ShardingCodecBuilder, ZstdCodec},
        ArrayBuilder, FillValue,
    },
    group::GroupBuilder,
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

impl ZarrWriter {
    pub fn init(save_path: &Path, array_path: &str) -> Result<(), ConsumerError> {
        let mut opts = FilesystemStoreOptions::default();
        opts.direct_io(true);
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new_with_options(save_path, opts).unwrap());

        let group = GroupBuilder::new().build(Arc::clone(&store), "/group")?;
        group.store_metadata()?;

        let shard_shape = vec![16, 512, 512];
        let inner_chunk_shape = vec![1, 512, 512];

        enum WriteMode {
            ShardedNoCompression,
            ShardedZstd,
            Zstd,
            Plain,
        }

        let mode = WriteMode::Plain;

        match mode {
            WriteMode::ShardedNoCompression => {
                let mut sharding_codec_builder =
                    ShardingCodecBuilder::new(inner_chunk_shape.as_slice().try_into()?);
                sharding_codec_builder.bytes_to_bytes_codecs(vec![]);

                let array = ArrayBuilder::new(
                    vec![65536, 512, 512],
                    zarrs::array::DataType::UInt16,
                    shard_shape.try_into().unwrap(),
                    FillValue::from(0u16),
                )
                .array_to_bytes_codec(Box::new(sharding_codec_builder.build()))
                .dimension_names(["i", "Ky", "Kx"].into())
                .build(Arc::clone(&store), array_path)?;

                array.store_metadata()?;
            }
            WriteMode::ShardedZstd => {
                let mut sharding_codec_builder =
                    ShardingCodecBuilder::new(inner_chunk_shape.as_slice().try_into()?);
                sharding_codec_builder
                    .bytes_to_bytes_codecs(vec![Box::new(ZstdCodec::new(1, true))]);

                let array = ArrayBuilder::new(
                    vec![65536, 512, 512],
                    zarrs::array::DataType::UInt16,
                    shard_shape.try_into().unwrap(),
                    FillValue::from(0u16),
                )
                .array_to_bytes_codec(Box::new(sharding_codec_builder.build()))
                .dimension_names(["i", "Ky", "Kx"].into())
                .build(Arc::clone(&store), array_path)?;

                array.store_metadata()?;
            }
            WriteMode::Plain => {
                let array = ArrayBuilder::new(
                    vec![65536, 512, 512],
                    zarrs::array::DataType::UInt16,
                    shard_shape.try_into().unwrap(),
                    FillValue::from(0u16),
                )
                // .array_to_bytes_codec(vec![])
                .dimension_names(["i", "Ky", "Kx"].into())
                .build(Arc::clone(&store), array_path)?;

                array.store_metadata()?;
            }
            WriteMode::Zstd => {
                let array = ArrayBuilder::new(
                    vec![65536, 512, 512],
                    zarrs::array::DataType::UInt16,
                    shard_shape.try_into().unwrap(),
                    FillValue::from(0u16),
                )
                .bytes_to_bytes_codecs(vec![Box::new(ZstdCodec::new(1, true))])
                .dimension_names(["i", "Ky", "Kx"].into())
                .build(Arc::clone(&store), array_path)?;

                array.store_metadata()?;
            }
        }

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

thread_local! {
    pub static PAGE_ALIGNED_BUF: RefCell<BytesMut> = RefCell::new(bytes_aligned(0));
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

        let buf = PAGE_ALIGNED_BUF.take();

        // FIXME: if len != chunk size, we need to buffer
        assert_eq!(len, 16);

        let size_bytes = len * (shape.0 * shape.1) as usize * size_of::<u16>();

        let mut buf = if buf.len() < size_bytes {
            let mut buf = bytes_aligned(size_bytes);
            buf.extend(iter::repeat(0).take(size_bytes));
            buf
        } else {
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
        buf = buf_frozen.try_into_mut().unwrap(); // FIXME: handle the case where the buffer is still in use?

        PAGE_ALIGNED_BUF.replace(buf);

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
    pub fn key_to_fspath(&self, key: &StoreKey) -> PathBuf {
        let mut path = self.save_path.clone();
        if !key.as_str().is_empty() {
            path.push(key.as_str().strip_prefix('/').unwrap_or(key.as_str()));
        }
        path
    }

    pub fn get_file_mutex(&self, key: &StoreKey) -> Arc<RwLock<()>> {
        let mut files = self.files.lock().unwrap();
        let file = files
            .entry(key.clone())
            .or_insert_with(|| Arc::new(RwLock::default()))
            .clone();
        drop(files);
        file
    }
}
