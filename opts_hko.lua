local log = loadfile("../helper/log.lua/log.lua")()

opt = {}
opt.onMac = true

if opt.onMac then
	opt.gpuflag = false
else
	opt.gpuflag = true
end

-- general options:
opt.dir     = 'of_record' -- subdirectory to save experiments in
opt.seed    = 1250         -- initial random seed
opt.imageDepth = 1

-- Model parameters:
opt.inputSizeW = 50   -- width of each input patch or image
opt.inputSizeH = 50  -- width of each input patch or image
opt.eta       = 1e-4 -- learning rate
opt.etaDecay  = 1e-5 -- learning rate decay
opt.momentum  = 0.9  -- gradient momentum
opt.maxEpoch  = 200 --max number of updates
opt.nSeq      = 20
opt.batchSize = 5
opt.input_nSeq = 5
opt.output_nSeq = 3
opt.width = opt.inputSizeW
opt.trainSamples = 2037 * 4
opt.validSamples = 2037 
opt.transf    = 2       -- number of parameters for transformation; 6 for affine or 3 for 2D transformation

opt.transfBetween = 4
opt.nFilters  = {4, 32}      --9,45} -- number of filters in the encoding/decoding layers
opt.nFiltersMemory   = {1, 17} --{45,60}

opt.fakeDepth = opt.batchSize
opt.maskStride = opt.imageH/opt.inputSizeH -- assume h = w

opt.kernelSize       = 3 -- size of kernels in encoder/decoder layers
opt.kernelSizeMemory = 3

--------------------------- training confirguration
opt.paraInit = 0.01
opt.parametersInitStd = 0.01
opt.trainIter = torch.floor(opt.trainSamples / opt.batchSize) 
opt.validIter = torch.floor(opt.validSamples / opt.batchSize) 
opt.selectStep = 2


--------------------------
opt.kernelSizeFlow   = 15
opt.padding   = torch.floor(opt.kernelSize / 2) -- pad input before convolutions
opt.dmin = -0.5 -- -0.5
opt.dmax = 0.5 -- 0.5
opt.gradClip = 50
opt.stride = 1 --opt.kernelSizeMemory -- no overlap
opt.constrWeight = {0, 1, 0.001}

opt.memorySizeW = opt.inputSizeW
opt.memorySizeH = opt.inputSizeH

opt.dataFile = '/mnt/disk/ankur/Viorica/data/mnist/dataset_fly_64x64_lines_train.t7'
opt.statInterval = 50 -- interval for printing error
pt.v            = false  -- be verbose
opt.display      = false -- true -- display stuff
opt.displayInterval = opt.statInterval*10
opt.save         = true -- save models

if not paths.dirp(opt.dir) then
   os.execute('mkdir -p ' .. opt.dir)
end
-- opt
opt = {}

opt.nSamples = 2037 * 4 -- number of frames in total
opt.nSeq = 20 -- length of the sequence in one trainSeq, input + output
opt.imageHeight = 100 -- number of rows and cols in one image matrix
opt.imageWidth = 100 
opt.imageDepth = 1
opt.dataFile = "../helper/trainData.txt"

opt.dataPath = "../data/hko/"

log.trace("@done load opt ")
