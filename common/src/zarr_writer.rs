use std::{cell::RefCell, marker::PhantomData, path::Path, sync::Arc};

use log::debug;
use ndarray::{Array3, Axis, Slice};
use num::cast::AsPrimitive;
use zarrs::{
    array::{
        codec::{array_to_bytes::sharding::ShardingCodecBuilder, ZstdCodec},
        ArrayBuilder, FillValue,
    },
    group::GroupBuilder,
    storage::{
        ReadableListableStorageTraits, ReadableWritableListableStorage,
        ReadableWritableListableStorageTraits,
    },
};

use crate::{
    consumer::{Consumer, ConsumerError},
    decoder::{Decoder, DecoderTargetPixelType},
    frame_stack::{FrameMeta, FrameStackHandle},
    zarr_dio::FilesystemStoreDIO,
};

pub struct ZarrWriter {}

impl ZarrWriter {
    pub fn init(save_path: &Path, array_path: &str) -> Result<(), ConsumerError> {
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStoreDIO::new(save_path).unwrap());

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
    buffer: Option<Array3<T>>,
    arr: zarrs::array::Array<dyn ReadableWritableListableStorageTraits>,
    _d: PhantomData<D>,
    _m: PhantomData<M>,
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
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStoreDIO::new(save_path).unwrap());
        let arr = zarrs::array::Array::open(Arc::clone(&store), array_path).unwrap();
        Self {
            arr,
            buffer: None,
            _d: Default::default(),
            _m: Default::default(),
        }
    }
}

impl<T, D, M> Consumer for ZarrChunkWriter<T, D, M>
where
    T: DecoderTargetPixelType + Send + Sync + zarrs::array::Element,
    u8: AsPrimitive<T>,
    u16: AsPrimitive<T>,
    D: Decoder<FrameMeta = M> + Send + Sync,
    M: FrameMeta + Send + Sync,
{
    fn consume_frame_stack(
        &mut self,
        handle: &FrameStackHandle<M>,
        decoder: &D,
        shm: &ipc_test::SharedSlabAllocator,
    ) -> Result<(), ConsumerError> {
        let len: usize = handle.len();
        let shape = handle.first_meta().get_shape();

        let mut buffer = match self.buffer.take() {
            None => Array3::from_shape_simple_fn([len, shape.0 as usize, shape.1 as usize], || {
                <T as num::Zero>::zero()
            }),
            Some(existing) => {
                if existing.shape().first().unwrap() < &len {
                    Array3::from_shape_simple_fn([len, shape.0 as usize, shape.1 as usize], || {
                        <T as num::Zero>::zero()
                    })
                } else {
                    existing
                }
            }
        };

        // FIXME: if len != chunk size, we need to buffer
        assert_eq!(len, 16);
        let mut view = buffer.view_mut();
        let mut output = view.slice_axis_mut(Axis(0), Slice::from(0..len));
        decoder.decode(shm, handle, &mut output, 0, len).unwrap();
        let elements = output.as_slice().unwrap();

        let chunk_indices = [(handle.first_meta().get_index() / len) as u64, 0u64, 0u64];

        self.arr
            .store_chunk_elements(&chunk_indices, elements)
            .unwrap();
        self.buffer = Some(buffer);
        Ok(())
    }

    type Decoder = D;
    type FrameMeta = M;

    fn finalize(&mut self) -> Result<(), ConsumerError> {
        Ok(())
    }
}
