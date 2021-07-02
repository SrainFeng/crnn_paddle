# import paddle
# import paddle.nn.functional as F
# import numpy as np

# x = [[[-2.0, 3.0, -4.0, 5.0],
#         [3.0, -4.0, 5.0, -6.0],
#         [-7.0, -8.0, 8.0, 9.0]],
#         [[1.0, -2.0, -3.0, 4.0],
#         [-5.0, 6.0, 7.0, -8.0],
#         [6.0, 7.0, 8.0, 9.0]]]
# x = paddle.to_tensor(x)
# out1 = F.log_softmax(x, axis=1)
# v = np.random.normal(loc=0.0, scale=0.02, size=x.shape).astype('float32')
#
# print(out1)
# print(x.size)
# print(v)

#
#
# a = paddle.ones([2, 3, 4])
# b = paddle.ones([2, 3, 4])
# b.stop_gradient = False
# c = paddle.ones([2, 3, 4])
# d = paddle.ones([2, 3, 4])
# d.stop_gradient = False
# ls = [a, b, c, d]
# # for i in ls:
# #     print(i.stop_gradient)
#
# print(a.stop_gradient)
# print(bool(1 - a.stop_gradient))

# example 1: dynamic graph
# emb = paddle.nn.Embedding(10, 10)
# layer_state_dict = emb.state_dict()
#
# # save state_dict of emb
# paddle.save(layer_state_dict, "emb.pdparams")
#
# scheduler = paddle.optimizer.lr.NoamDecay(
#     d_model=0.01, warmup_steps=100, verbose=True)
# adam = paddle.optimizer.Adam(
#     learning_rate=scheduler,
#     parameters=emb.parameters())
# opt_state_dict = adam.state_dict()
#
# # save state_dict of optimizer
# paddle.save(opt_state_dict, "adam.pdopt")
# # save weight of emb
# # paddle.save(emb.weight, "emb.weight.pdtensor")
#
# # load state_dict of emb
# load_layer_state_dict = paddle.load("emb.pdparams")
# # load state_dict of optimizer
# load_opt_state_dict = paddle.load("adam.pdopt")
# # load weight of emb
# # load_weight = paddle.load("emb.weight.pdtensor")
#
# for k, v in load_layer_state_dict.items():
#     print(k, v)
# model_state_file = "output/checkpoints/model.pdparams"
# checkpoint = paddle.load(model_state_file)
# for k, v in checkpoint.items():
#     print(k)
#     print(v.shape)
# paddle.utils.run_check()
# data = [[5,8,9,5],
#         [0,0,1,7],
#         [6,9,2,4]]
#
# x = paddle.to_tensor(data)
# print(x.argmax(axis=1))

print("try: {:0<20}".format("\u80fd"))
print('try2: %-20s' % "\u73b0")
