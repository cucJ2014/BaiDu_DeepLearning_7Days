import paddle
import paddle.fluid as fluid
import paddleslim as slim
import numpy as np
import paddle.dataset.mnist as reader
import paddleslim.quant as quant

use_gpu = fluid.is_compiled_with_cuda()
exe, train_program, val_program, inputs, outputs = slim.models.image_classification("MobileNet", [1, 28, 28], 10, use_gpu=use_gpu)
place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()


train_reader = paddle.batch(
        reader.train(), batch_size=128, drop_last=True)
test_reader = paddle.batch(
        reader.test(), batch_size=128, drop_last=True)
data_feeder = fluid.DataFeeder(inputs, place)

def train(prog):
    iter = 0
    loss_list=[]
    for data in train_reader():
        acc1, acc5, loss = exe.run(prog, feed=data_feeder.feed(data), fetch_list=outputs)
        loss_list.append(loss)
        if iter % 100 == 0:
            print('train iter={}, top1={}, top5={}, loss={}'.format(iter, acc1.mean(), acc5.mean(), loss.mean()))
        iter += 1
    avg_los = np.mean(loss_list)

def test(prog):
    iter = 0
    res = [[], []]
    for data in test_reader():
        acc1, acc5, loss = exe.run(prog, feed=data_feeder.feed(data), fetch_list=outputs)
        if iter % 100 == 0:
            print('test iter={}, top1={}, top5={}, loss={}'.format(iter, acc1.mean(), acc5.mean(), loss.mean()))
        res[0].append(acc1.mean())
        res[1].append(acc5.mean())
        iter += 1
    print('final test result top1={}, top5={}'.format(np.array(res[0]).mean(), np.array(res[1]).mean()))

train(train_program)

test(val_program)

# 量化模型
place = exe.place
config = { 'weight_quantize_type': 'channel_wise_abs_max', 'activation_quantize_type': 'moving_average_abs_max', 'quantize_op_types': ['depthwise_conv2d', 'mul', 'conv2d'] }
quant_program = quant.quant_aware(train_program, place, config, for_test=False)    #请在次数添加你的代码
val_quant_program = quant.quant_aware(val_program, place, config, for_test=True)    #请在次数添加你的代码
train(quant_program)
test(val_quant_program)

