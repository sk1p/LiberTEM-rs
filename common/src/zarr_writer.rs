use std::{marker::PhantomData, path::Path, sync::Arc};

use log::debug;
use ndarray::{Array3, Axis, Slice};
use num::cast::AsPrimitive;
use zarrs::{
    array::{
        codec::{array_to_bytes::sharding::ShardingCodecBuilder, ZstdCodec},
        ArrayBuilder, FillValue,
    },
    group::GroupBuilder,
    storage::{store, ReadableWritableListableStorage},
};

use crate::{
    consumer::{Consumer, ConsumerError},
    decoder::{Decoder, DecoderTargetPixelType},
    frame_stack::{FrameMeta, FrameStackHandle},
};

pub struct ZarrWriter {}

impl ZarrWriter {
    pub fn init(save_path: &Path, array_path: &str) -> Result<(), ConsumerError> {
        let store: ReadableWritableListableStorage =
            Arc::new(store::FilesystemStore::new(save_path).unwrap());

        let group = GroupBuilder::new().build(Arc::clone(&store), "/group")?;
        group.store_metadata()?;

        let shard_shape = vec![128, 512, 512];
        let inner_chunk_shape = vec![1, 512, 512];

        if false {
            let mut sharding_codec_builder =
                ShardingCodecBuilder::new(inner_chunk_shape.as_slice().try_into()?);
            sharding_codec_builder.bytes_to_bytes_codecs(vec![Box::new(ZstdCodec::new(5, true))]);

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
        } else {
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
    buffer: Array3<T>,
    store: ReadableWritableListableStorage,
    cursor: usize,
    array_path: String,
    chunk_indices: Vec<u64>,
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
    pub fn new(partition_chunk_indices: &[u64], save_path: &Path, array_path: &str) -> Self {
        let store: ReadableWritableListableStorage =
            Arc::new(store::FilesystemStore::new(save_path).unwrap());
        let arr = zarrs::array::Array::open(Arc::clone(&store), array_path).unwrap();
        let chunk_grid = arr.chunk_grid();
        let chunk_shape: Vec<std::num::NonZero<u64>> = chunk_grid
            .chunk_shape(partition_chunk_indices, arr.shape())
            .unwrap()
            .unwrap()
            .to_vec();
        assert!(chunk_shape.len() == 3);

        // FIXME: ok, so allocating a partition-sized buffer here is
        // understandably quite costly, and

        Self {
            store,
            buffer: Array3::from_shape_simple_fn(
                [
                    chunk_shape[0].get() as usize,
                    chunk_shape[1].get() as usize,
                    chunk_shape[2].get() as usize,
                ],
                || <T as num::Zero>::zero(),
            ),
            cursor: 0,
            array_path: array_path.to_owned(),
            chunk_indices: Vec::from(partition_chunk_indices),
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
        let mut view = self.buffer.view_mut();
        let mut output = view.slice_axis_mut(Axis(0), Slice::from(self.cursor..self.cursor + len));
        decoder.decode(shm, handle, &mut output, 0, len).unwrap();
        self.cursor += len;
        Ok(())
    }

    type Decoder = D;
    type FrameMeta = M;

    fn finalize(&mut self) -> Result<(), ConsumerError> {
        let arr = zarrs::array::Array::open(Arc::clone(&self.store), &self.array_path)?;
        let elements = self.buffer.as_slice().unwrap();
        arr.store_chunk_elements(&self.chunk_indices, elements)
            .unwrap();

        Ok(())
    }
}
