use ipc_test::SharedSlabAllocator;

use crate::{
    decoder::{Decoder, DecoderTargetPixelType},
    frame_stack::{FrameMeta, FrameStackHandle},
    generic_connection::AcquisitionConfig,
};

pub trait DecoderService {
    fn decode(&self);
}

pub type ConsumerError = Box<dyn std::error::Error + Sync + Send + 'static>;

pub trait Consumer: Send + Sync {
    type Decoder: Decoder;
    type FrameMeta: FrameMeta;

    fn consume_frame_stack(
        &mut self,
        handle: &FrameStackHandle<Self::FrameMeta>,
        decoder: &Self::Decoder,
        shm: &SharedSlabAllocator,
    ) -> Result<(), ConsumerError>;

    fn finalize(&mut self) -> Result<(), ConsumerError>;
}

#[cfg(test)]
mod test {
    use std::{marker::PhantomData, path::PathBuf};

    use crate::{
        consumer::{Consumer, ConsumerError},
        decoder::{Decoder, DecoderTargetPixelType},
        frame_stack::{FrameMeta, FrameStackHandle},
        generic_cam_client::GenericCamClient,
    };
    use ndarray::{Array3, Axis, Slice};
    use num::{cast::AsPrimitive, Zero};
    use serde::{Deserialize, Serialize};
    use tempfile::{tempdir, TempDir};

    fn get_socket_path() -> (TempDir, PathBuf) {
        let socket_dir = tempdir().unwrap();
        let socket_as_path = socket_dir.path().join("stuff.socket");

        (socket_dir, socket_as_path)
    }

    #[test]
    fn test_example_usage() {
        #[derive(Debug, Serialize, Deserialize, Clone)]
        struct MyFrameMeta {}

        impl FrameMeta for MyFrameMeta {
            fn get_data_length_bytes(&self) -> usize {
                todo!()
            }

            fn get_dtype_string(&self) -> String {
                todo!()
            }

            fn get_shape(&self) -> (u64, u64) {
                todo!()
            }

            fn get_index(&self) -> usize {
                todo!()
            }
        }

        #[derive(Default)]
        struct NoopDecoder {}
        impl Decoder for NoopDecoder {
            type FrameMeta = MyFrameMeta;

            fn decode<T>(
                &self,
                shm: &ipc_test::SharedSlabAllocator,
                input: &crate::frame_stack::FrameStackHandle<Self::FrameMeta>,
                output: &mut ndarray::ArrayViewMut3<'_, T>,
                start_idx: usize,
                end_idx: usize,
            ) -> Result<(), crate::decoder::DecoderError>
            where
                T: crate::decoder::DecoderTargetPixelType,
                u8: num::cast::AsPrimitive<T>,
                u16: num::cast::AsPrimitive<T>,
            {
                Ok(())
            }

            fn zero_copy_available(
                &self,
                handle: &crate::frame_stack::FrameStackHandle<Self::FrameMeta>,
            ) -> Result<bool, crate::decoder::DecoderError> {
                Ok(false)
            }
        }
        let (_socket_dir, socket_as_path) = get_socket_path();
        let mut cam_client: GenericCamClient<NoopDecoder> =
            GenericCamClient::new(&socket_as_path.to_str().unwrap()).unwrap();

        struct MyConsumer<T, D, M>
        where
            T: DecoderTargetPixelType,
            u8: AsPrimitive<T>,
            u16: AsPrimitive<T>,
            D: Decoder + Send + Sync,
            M: FrameMeta + Send + Sync,
        {
            buffer: Array3<T>,
            cursor: usize,
            _d: PhantomData<D>,
            _m: PhantomData<M>,
        }

        impl<T, D, M> MyConsumer<T, D, M>
        where
            T: DecoderTargetPixelType,
            u8: AsPrimitive<T>,
            u16: AsPrimitive<T>,
            D: Decoder + Send + Sync,
            M: FrameMeta + Send + Sync,
        {
            fn new(partition_size: (usize, usize, usize)) -> Self {
                Self {
                    buffer: Array3::from_shape_simple_fn(partition_size, || <T as Zero>::zero()),
                    cursor: 0,
                    _d: Default::default(),
                    _m: Default::default(),
                }
            }
        }

        impl<T, D, M> Consumer for MyConsumer<T, D, M>
        where
            T: DecoderTargetPixelType + Send + Sync,
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
                let mut output =
                    view.slice_axis_mut(Axis(0), Slice::from(self.cursor..self.cursor + len));

                decoder.decode(shm, handle, &mut output, 0, len).unwrap();

                Ok(())
            }

            type Decoder = D;
            type FrameMeta = M;

            fn finalize(&mut self) -> Result<(), ConsumerError> {
                todo!("we were only decoding into a buffer, which we can now do something with")
            }
        }

        // here we specify f32 as dtype; in real code this would need to dispatch on dtype
        let consumer: MyConsumer<f32, NoopDecoder, MyFrameMeta> = MyConsumer::new((25, 512, 512));

        cam_client.register_consumer(Box::new(consumer)).unwrap();
    }
}
