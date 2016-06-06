import sys
import os
import shutil
from subprocess import call

def moviefy(input_folder, output_folder):

    framerate = 8
    
    input_folder = input_folder.strip('/') + '/'
    output_folder = output_folder.strip('/') + '/'

    namelist = ['AG_', 'bca_', 'es_', 
            'forest', 'influence_cropped_b_', 
            'npp_', 'pop_grad_', 'rain_', 
            'soil_deg_', 'trade_network_', 
            'waterflow_']

    for name in namelist:
        input_string = input_folder + name + "%3d.png"
        output_string = output_folder + name.strip('_') + '.mp4'
        call(["ffmpeg", "-r", `framerate`, "-i", input_string, output_string])

if __name__ == '__main__':

    print sys.argv

    if len(sys.argv) < 3:
        print 'usage is python moviefy.py [input folder] [output folder]'
    
    
    if os.path.exists(sys.argv[2]):
        shutil.rmtree(sys.argv[2])
    os.makedirs(sys.argv[2])
    moviefy(sys.argv[1], sys.argv[2])


