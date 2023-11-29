import torch
from . import initialization as init


class SegmentationModel(torch.nn.Module):
    def initialize(self):
        init.initialize_decoder(self.decoder1)
        init.initialize_decoder(self.decoder2)
        init.initialize_decoder(self.decoder3)
        init.initialize_decoder(self.decoder4)
        init.initialize_head(self.segmentation_head1)
        init.initialize_head(self.segmentation_head2)
        init.initialize_head(self.segmentation_head3)
        init.initialize_head(self.segmentation_head4)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        self.check_input_shape(x)

        features = self.encoder(x)
        # index value from npymax.py, each represents the maximum value for each lesion
        CMBindex = 12970
        WHMindex = 5061
        epvsindex = 12174
        lacuneindex = 5061
        
        
        if x[0, 0, 0, 0] * CMBindex == 1:
            print("1")
            decoder_output1 = self.decoder1(*features)

            masks1 = self.segmentation_head1(decoder_output1)

            if self.classification_head is not None:
                labels1 = self.classification_head(features[-1])
                return masks1, labels1

            return masks1
        elif x[0, 0, 0, 0] * WHMindex == 2:
            print("2")
            decoder_output2 = self.decoder2(*features)

            masks2 = self.segmentation_head2(decoder_output2)

            if self.classification_head is not None:
                labels2 = self.classification_head(features[-1])
                return masks2, labels2

            return masks2
        elif x[0, 0, 0, 0] * epvsindex == 3:
            print("3")
            decoder_output3 = self.decoder3(*features)

            masks3 = self.segmentation_head3(decoder_output3)

            if self.classification_head is not None:
                labels3 = self.classification_head(features[-1])
                return masks3, labels3

            return masks3
        elif x[0, 0, 0, 0] * lacuneindex == 4:
            print("4")
            decoder_output4 = self.decoder4(*features)

            masks4 = self.segmentation_head4(decoder_output4)

            if self.classification_head is not None:
                labels4 = self.classification_head(features[-1])
                return masks4, labels4

            return masks4

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x
