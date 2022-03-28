import torch

# x = torch.randn(3, requires_grad=True)
# print(x)
# y=x+2
# print(y)
# z=y*y*2
# print(z)
# # z=z.mean()
# print(z)
# v =torch.tensor(data=[-1, 0, 1], dtype=torch.float32)
# z.backward(v)
# print(x.grad)


# a = torch.randn(3,2, requires_grad=True)
# print(a)
# # a.requires_grad_(False)
# #y = a.detach() + 1
# with torch.no_grad():
#     b = a*5
#     print(b)

weights = torch.ones(4, dtype=torch.float32, requires_grad=True)
print(weights)
for epoch in range(16):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()