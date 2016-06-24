
local log = loadfile("log.lua")()

opt = {}
opt.onMac = true

-- general options:
opt.imageDepth = 1

opt.useSeqLSTM = true

-- Model parameters:
opt.inputSizeW = 100   -- width of each input patch or image
opt.inputSizeH = 100  -- width of each input patch or image
opt.nFiltersMemory   = {1, 17}  --{45,60}

opt.memorySizeW = opt.inputSizeW
opt.memorySizeH = opt.inputSizeH
opt.batchSize = 2

opt.inputSeqLen = 2
opt.outputSeqLen = 3
opt.totalSeqLen = opt.inputSeqLen + opt.outputSeqLen
opt.encoderRho = 1
opt.decoderRho = opt.outputSeqLen

-- number of parameters for transformation; 6 for affine or 3 for 2D transformation
-- model confirguration
opt.kernelSize       = 3 -- size of kernels in encoder/decoder layers
opt.padding   = torch.floor(opt.kernelSize / 2) -- pad input before convolutions
opt.stride= 1

-- training confirguration
opt.maxIter = 100
opt.lr = 0.01
opt.saveDir = '/mnt/ficusengland/ssd1/xhzeng2/hko/'
opt.save         = true -- save models
opt.nSamples = 2037 * 4 -- number of frames in total
opt.nSeq = 20 -- length of the sequence in one trainSeq, input + output
opt.imageHeight = 100 -- number of rows and cols in one image matrix
opt.imageWidth = 100 
opt.imageDepth = 1
opt.listFile = "../all_data_list.txt"
opt.dataPath = "../"

-- if not paths.dirp(opt.dir) then
--        os.execute('mkdir -p ' .. opt.dir)
-- end
-- confirguration
opt.seed = 1
