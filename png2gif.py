
def make_gif(time, iter):
    import glob
    from PIL import Image, ImageChops
    import numpy
    from PIL.GifImagePlugin import getheader, getdata
    import gifmaker
    sequence = []
    # fileList = glob.glob('../hko_fast/1900it6850outputn_*')
    fileList = []

    # iter = 24000


    for i in range(15):
        name = '../hko_lstm_baseline/' + str(time) + 'it' + str(iter) + 'outputn_' + str(i + 1) + '.png'
        fileList.append(name)
    print(fileList)

    for file in fileList:
        print(file)
        im = Image.open(file)
        img = numpy.asarray(im)
        print(img)
        sequence.append(im)
    fp = open(str(iter) + "out.gif", "wb")
    gifmaker.makedelta(fp, sequence)
    fp.close()
    # gifmaker.save_gif(sequence, 'input.gif')

