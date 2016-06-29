dofile 'model_hko_flow.lua'
a = torch.Tensor(1, 1, 100, 100):cuda()
print(model)
out = model:updateOutput(a)

out2 = branch_memory:updateOutput(a)
