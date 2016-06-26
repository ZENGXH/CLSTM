py = require 'python'

make_gif =  py.import "png2gif".make_gif
iter = 24000
time =  0300
make_gif(time, iter)

