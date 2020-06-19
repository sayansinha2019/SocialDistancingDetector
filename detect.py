import numpy as np
import time
import argparse
import cv2

import os
import sys
py_dll_path = os.path.join(sys.exec_prefix, 'Library', 'bin')
os.environ['PATH'] += py_dll_path



def script_method(fn, _rcb=None):
	return fn
def script(obj, optimize=True, _frames_up=0, _rcb=None):
	return obj    
import torch.jit
torch.jit.script_method = script_method 
torch.jit.script = script




import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
import imageio
import skimage
assert torch.__version__.split('.')[0] == '1'
print('CUDA available: {}'.format(torch.cuda.is_available()))

from threading import Thread
import cv2




class VideoGet:
   
	def __init__(self, src=0):
		self.stream = cv2.VideoCapture(src)
		self.stream.set(4,1024)
		self.grabbed, self.frame = self.stream.read()
		self.frame = cv2.resize(self.frame, (640,480))
		self.stopped = False


	def start(self):    
		Thread(target=self.get, args=()).start()
		return self

	def get(self):
		while not self.stopped:
			if not self.grabbed:
				self.stop()
			else:
				self.grabbed, self.frame = self.stream.read()
				self.frame = cv2.resize(self.frame, (640,480))

	def stop(self):
		self.stopped = True
		self.stream.release()

def draw_caption(image, box, caption):
	caption = str(caption)
	b = np.array(box).astype(int)
	# cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)


def padImage(image):
	rows, cols, cns = image.shape

	pad_w = 32 - rows%32
	pad_h = 32 - cols%32

	new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(image.dtype)
	new_image[:rows, :cols, :] = image.astype(np.float32)
	return new_image

label_class_name = {0:'person'}

color_label = {0 : (255,0,0), 
            }

alpha = 0.0
# foreground = cv2.imread("/home/mudit/Downloads/cloudindex.jpeg")
# # foreground = foreground[30:200, 30:200, :]
# foreground = cv2.resize(foreground, (480,180))



import math
def distance(p0, p1):
	return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)



def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default='csv')
	parser.add_argument('--coco_path', help='Path to COCO directory', default='cocodataset')
	parser.add_argument('--model_path', help='Path to model (.pt) file.', type=str, default='coco_resnet_50_map_0_335_state_dict.pt')
	parser = parser.parse_args(args)


    # Create the model

	parser.model_path = "social_distancing.pt"
	retinanet = torch.load(parser.model_path, map_location=torch.device('cuda'))
    # retinanet = torch.load(parser.model_path)
    # retinanet = torch.load(parser.model_path, map_location=torch.device('cpu'))
    # retinanet = torch.load(parser.model_path)#, map_location=torch.device('cpu'))
    # retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    # retinanet.load_state_dict(torch.load(parser.model_path))
    
	use_gpu = True

    
	if use_gpu:
		device = torch.device('cuda')
		retinanet.cuda()
	else:
		device = torch.device('cpu')
		retinanet.cpu()
    # device = torch.device('cpu')
	retinanet = retinanet.to(device)        


	retinanet.eval()


	transformer=transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


   

	try:
		cap = cv2.VideoCapture('social_distancing_test.mp4')
	except:
		cap = cv2.VideoCapture(camera_url)
	frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # try:
    #     video_getter = VideoGet(int(camera_url)).start()
    # except:
    #     video_getter = VideoGet(camera_url).start()

	video_preview = input("Want to see Video Preview : (y/n)")
    # source = "rtsp://169.254.1.1:554/live/0/MAIN/"
    # camera_url = "http://77.22.202.109:80/mjpg/video.mjpg"
 #    try:
    #     video_getter = VideoGet(int(camera_url)).start()
    # except:
    #     video_getter = VideoGet(camera_url).start()
	frame_id = 0
    # video_out = "res_vid/output_{}.mp4".format(time.time())
    # video_writer = cv2.VideoWriter(video_out,
    #                                cv2.VideoWriter_fourcc(*'XVID'), 
    #                                1.0, 
    #                                (frame_w, frame_h))


	while True:
		try:
            # im = imageio.imread(imagepath)
			ret, im = cap.read()
            # im = video_getter.frame
			im = cv2.resize(im, (640,480))

            # im = cv2.imread(imagepath)
            # if frame_id % 10 !=0:
            #     frame_id = frame_id + 1
            #     continue
			frame_id = frame_id + 1
			if frame_id % 5 != 1:
				continue
			org_im = im.copy()
			print (org_im.shape)
            
            # added_image = cv2.addWeighted(org_im[150:330,150:630,:],alpha,foreground,1-alpha,0)
            # org_im[100:280,100:580] = added_image
            # org_im[-180:,-480:] = added_image
            # cv2.putText(org_im, "www.cloud9securities.in", (0, 1000), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, ), 4)
            # added_image = cv2.addWeighted(org_im[150:350,150:350,:],alpha,foreground,1-alpha,0)
            # org_im[200:400,200:400] = added_image
            
			im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            #im = skimage.transform.resize(im, (640, 928))
            #im = skimage.transform.resize(im, (1008, 928))
			im=padImage(im)
			img = torch.from_numpy(im).float().permute(2, 0, 1)/255        
			img = transformer(img).unsqueeze(dim=0)
            
			with torch.no_grad():
				st = time.time()
				print('processing...')
				scores, classification, transformed_anchors = retinanet(img.float().to(device))
				idxs = np.where(scores.cpu()>0.5)
				# img = cv2.cvtColor((255*im).astype(np.uint8), cv2.COLOR_BGR2RGB)
				# print (scores.cpu())
				box_center_list = []
				person_count = 0
        
				for j in range(idxs[0].shape[0]):
					label_name = int(classification[idxs[0][j]])
					if label_class_name[label_name] =='Mask':
						continue
                    # print (scores.cpu()[idxs])
					bbox = transformed_anchors[idxs[0][j], :]
					x1 = int(bbox[0])
					y1 = int(bbox[1])
					x2 = int(bbox[2])
					y2 = int(bbox[3])
                    # label_name = dataset_val.labels[int(classification[idxs[0][j]])]
					x_center , y_center = int((x1+x2)/2), y2#int((y1+y2)/2)
					box_center = (x_center, y_center)
					box_center_list.append(box_center)
                    # draw_caption(org_im, (x1, y1, x2, y2), label_class_name[label_name])
                    
					cv2.rectangle(org_im, (x1, y1), (x2, y2), color=color_label[label_name], thickness=1)
                    # print(label_name)
                    
				for first_center in box_center_list:
					for sec_center in box_center_list:
						if first_center == sec_center:
							continue
						else:
							line_distance =distance(first_center, sec_center)
							if line_distance < 50:
								person_count = person_count+1
								org_im = cv2.line(org_im, first_center, sec_center, (0,0,255), 1)
								cv2.rectangle(org_im, (x1, y1), (x2, y2), color=(0,0,255), thickness=1)
							else:
								cv2.rectangle(org_im, (x1, y1), (x2, y2), color=color_label[label_name], thickness=1)
                            # if line_distance > 30 and line_distance < 80:
                            #     org_im = cv2.line(org_im, first_center, sec_center, (12,127,0), 1)
				caption = "Distance Violation(s): {}".format(str(int(person_count/2)))
				if (int(person_count/2)) >0:
					cv2.putText(org_im, caption, (100, 100 - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
					cv2.imwrite('social_distancing_violations/{}.png'.format(frame_id), org_im)
                # video_writer.write(np.uint8(org_im))
				if video_preview =='y':
					cv2.imshow('img', org_im)

		except Exception as E:
			print (E)
			ass
		if cv2.waitKey(1) & 0xFF == ord('q'):
            # video_getter.stop()
			cv2.destroyAllWindows()
            # video_writer.release()
			break
		print('Elapsed time: {}'.format(time.time()-st))

if __name__ == '__main__':
	main()