import nnabla as nn

def enable_gpu():
    import nnabla_ext.cuda
    import nnabla_ext.cudnn

    from nnabla.ext_utils import get_extension_context

    ctx = get_extension_context('cudnn', device_id='0')
    nn.set_default_context(ctx)

if __name__ == "__main__":
    enable_gpu()
    nnabla_ext.cudnn.check_gpu('0')
