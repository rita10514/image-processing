from matplotlib import pyplot as plt
from scipy import signal
from time import time
import numpy as np
import winsound
import cv2


#helping functions


def imReadAndConvert(filename, mode = 0):
	data = cv2.imread(filename)
	if mode != 0:
		data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
	if data.ndim == 2: return data
	if mode == 0:
		if len(data[0][0]) == 3: return np.round(data @ [0.299, 0.587, 0.114])
		if len(data[0][0]) == 4: return np.round(data @ [0.299, 0.587, 0.114, 0])
	return data

def imDisplay(image):
	plt.imshow(image, cmap='gray')
	plt.show()

def rgb_gray(img):
	if img.ndim == 3:
	   return np.round(img @ [0.299, 0.587, 0.114])
	else: return img   

def bgr_gray(img):
	if img.ndim == 3:
	   return np.round(img @ [0.114, 0.587, 0.299])
	else: return img
	
def gray_to_3ch(img):
	h, w = img.shape
	res = np.zeros((h, w, 3))
	for i in range(3):
		res[:,:,i] = img
	return res	

def pad_img(img, edge, mirror = 0):
	pdd_shape = list(img.shape)
	pdd_shape[0] += 2 * edge ; 	pdd_shape[1] += 2 * edge
	h, w = img.shape[:2] ; H, W = pdd_shape[:2]
	pdd_img = np.zeros(pdd_shape)
	pdd_img[edge : H - edge, edge : W - edge] = img
	if mirror != 0:
		pdd_img[:edge, :edge] = img[edge:0:-1, edge:0:-1]
		pdd_img[:edge, (W - edge):] = img[edge:0:-1, w:(w - edge - 1):-1]
		pdd_img[(H - edge):, (W - edge):] = img[h:(h - edge - 1):-1, w:(w - edge - 1):-1]
		pdd_img[(H - edge):, :edge] = img[h:(h - edge - 1):-1, edge:0:-1]
		pdd_img[edge:(H - edge), :edge] = img[:, edge:0:-1]
		pdd_img[edge:(H - edge), (W - edge):] = img[:, w:(w - edge - 1):-1]
		pdd_img[:edge, edge:(W - edge)] = img[edge:0:-1, :]
		pdd_img[(H - edge):, edge:(W - edge)] = img[h:(h - edge - 1):-1, :]
	return pdd_img
	
def sub_img(img, cp, size, mirror = 0):
	c_w, c_h = cp
	h, w = img.shape
	edge = int(size / 2)
	res = np.zeros((size, size))
	rng = range(-edge, edge + 1)
	for i in rng:
		for j in rng:
			if mirror == 0:
				in_rng = ((i + c_h) in range(h))&((j + c_w) in range(w))
				if in_rng:
					res[i + edge,j + edge] = img[i + c_h, j + c_w]
			else:
				I,J = abs(h - 1 - abs(i + c_h - h + 1)), abs(w - 1 - abs(j + c_w - w + 1))
				res[i + edge,j + edge] = img[I,J]
	return res

def gauss_ker(sig):
	edge = int(round(2 * sig))
	size = 2 * edge + 1
	rng = range(-edge, edge + 1)
	d = np.array([[(i * i + j * j) ** 0.5 for j in rng] for i in rng])
	ker = (np.exp(-(d / sig) ** 2)) / (np.pi * (sig ** 2))
	ker = ker / sum(ker.ravel())
	return ker

def blurImage(inImage,sigma):
	return conv2D(inImage, gauss_ker(sigma)) 

def conv2D(img, ker, mode = 0):
	edge = int(len(ker) / 2)
	padded = pad_img(img, edge, mirror = 1)
	if mode != 0:
		padded = pad_img(padded, edge)
	new_shape = tuple(np.subtract(padded.shape, ker.shape) + 1) + ker.shape
	sub_mats = np.lib.stride_tricks.as_strided(padded, new_shape, (2 * padded.strides))
	return np.einsum('ij,klij->kl', ker, sub_mats)

def display_two(im1, im2):
	plt.subplot(1,2,1); plt.imshow(im1, cmap='gray')
	plt.subplot(1,2,2); plt.imshow(im2, cmap='gray')
	plt.show()

def PolyArea(x,y):
	return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y ,np.roll(x, 1)))

#canny - edge detection

def Normalize(img):
	img = bgr_gray(img)
	min_val = np.min(img.ravel())
	max_val = np.max(img.ravel())
	output = (img.astype('float') - min_val) / (max_val - min_val)
	return output

def DerivativesOfGaussian(img, sigma):
	
	# sobel kernel
	Sx = np.array([ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1] ])
	Sy = np.array([ [-1, -2, -1], [0, 0, 0], [1, 2, 1] ])

	# gaussian kernel
	halfSize = 3 * sigma
	maskSize = 2 * halfSize + 1 
	mat = np.ones((maskSize,maskSize)) / (float)( 2 * np.pi * (sigma**2))
	xyRange = np.arange(-halfSize, halfSize+1)
	xx, yy = np.meshgrid(xyRange, xyRange)	
	x2y2 = (xx**2 + yy**2)	
	exp_part = np.exp(-(x2y2/(2.0*(sigma**2))))
	gSig = mat * exp_part
	
	# convolve to create gx & gy
	gx = signal.convolve2d(Sx, gSig)			  # Convolution
	gy = signal.convolve2d(Sy, gSig)
		
	# apply kernels for Ix & Iy
	Ix = conv2D(255 * img, gx)
	Iy = conv2D(255 * img, gy)
	
	
	return Ix, Iy
	
def MagAndOrientation(Ix, Iy, t_low):
	
	# compute magnitude
	mag = np.sqrt(Ix**2 + Iy**2)
	
	# normalize magnitude image
	normMag = Normalize(mag)
	
	# compute orientation of gradient
	orient = np.arctan2(Iy, Ix)
	
	# round elements of orient
	orientRows = orient.shape[0]
	orientCols = orient.shape[1]

	for i in range(0, orientRows):
		for j in range(0, orientCols):
			if normMag[i,j] > t_low:
				# case 0
				if (orient[i,j] > (- np.pi / 8) and orient[i,j] <= (np.pi / 8)):
					orient[i,j] = 0
				elif (orient[i,j] > (7 * np.pi / 8) and orient[i,j] <= np.pi):
					orient[i,j] = 0
				elif (orient[i,j] >= -np.pi and orient[i,j] < (-7 * np.pi / 8)):
					orient[i,j] = 0
				# case 1
				elif (orient[i,j] > (np.pi / 8) and orient[i,j] <= (3 * np.pi / 8)):
					orient[i,j] = 3
				elif (orient[i,j] >= (-7 * np.pi / 8) and orient[i,j] < (-5 * np.pi / 8)):
					orient[i,j] = 3
				# case 2
				elif (orient[i,j] > (3 * np.pi / 8) and orient[i,j] <= (5 * np.pi /8)):
					orient[i,j] = 2
				elif (orient[i,j] >= (-5 * np.pi / 4) and orient[i,j] < (-3 * np.pi / 8)):
					orient[i,j] = 2
				# case 3
				elif (orient[i,j] > (5 * np.pi/8) and orient[i,j] <= (7 * np.pi /8)):
					orient[i,j] = 1
				elif (orient[i,j] >= (-3 * np.pi / 8) and orient[i,j] < (-np.pi / 8)):
					orient[i,j] = 1

	# convert orientation to color
	orientColor = orient.astype(np.uint8)
	orientColor = cv2.cvtColor(orientColor, cv2.COLOR_GRAY2BGR)
	for i in range(0, orientRows):
		for j in range(0, orientCols):
			if normMag[i,j] > t_low:
				if (orient[i,j] == 0):
					orientColor[i,j] = [0, 0, 255]
				elif (orient[i,j] == 1):
					orientColor[i,j] = [0, 255, 0]
				elif (orient[i,j] == 2):
					orientColor[i,j] = [255, 0, 0]
				elif (orient[i,j] == 3):
					orientColor[i,j] = [220, 220, 220]
	
	return normMag, orient
	  
def NMS(mag, orient, t_low):
	mag_thin = np.zeros(mag.shape)
	for i in range(mag.shape[0] - 1):
		for j in range(mag.shape[1] - 1):
			if mag[i][j] < t_low:
				continue
			if orient[i][j] == 0:
				if mag[i][j] > mag[i][j-1] and mag[i][j] >= mag[i][j+1]:
					mag_thin[i][j] = mag[i][j]
			if orient[i][j] == 1:
				if mag[i][j] > mag[i-1][j+1] and mag[i][j] >= mag[i+1][j-1]:
					mag_thin[i][j] = mag[i][j]
			if orient[i][j] == 2:
				if mag[i][j] > mag[i-1][j] and mag[i][j] >= mag[i+1][j]:
					mag_thin[i][j] = mag[i][j]
			if orient[i][j] == 3:
				if mag[i][j] > mag[i-1][j-1] and mag[i][j] >= mag[i+1][j+1]:
					mag_thin[i][j] = mag[i][j]

	
	return mag_thin
	
def linking(mag_thin, orient, tLow, tHigh):
	result_binary = np.zeros(mag_thin.shape)
	
	# forward scan
	for i in range(0, mag_thin.shape[0] - 1):		   # rows
		for j in range(0, mag_thin.shape[1] - 1):	   # columns
			if mag_thin[i][j] >= tHigh:
				if mag_thin[i][j+1] >= tLow:			# right
					mag_thin[i][j+1] = tHigh
				if mag_thin[i+1][j+1] >= tLow:		  # bottom right
					mag_thin[i+1][j+1] = tHigh
				if mag_thin[i+1][j] >= tLow:			# bottom
					mag_thin[i+1][j] = tHigh
				if mag_thin[i+1][j-1] >= tLow:		  # bottom left
					mag_thin[i+1][j-1] = tHigh
	
	# backwards scan - CHANGED TO -2
	for i in range(mag_thin.shape[0] - 2, 0, -1):	   # rows
		for j in range(mag_thin.shape[1] - 2, 0, -1):   # columns
			if mag_thin[i][j] >= tHigh:
				if mag_thin[i][j-1] > tLow:			 # left
					mag_thin[i][j-1] = tHigh
				if mag_thin[i-1][j-1]:				  # top left
					mag_thin[i-1][j-1] = tHigh
				if mag_thin[i-1][j] > tLow:			 # top
					mag_thin[i-1][j] = tHigh
				if mag_thin[i-1][j+1] > tLow:		   # top right
					mag_thin[i-1][j+1] = tHigh

	# fill in result_binary
	for i in range(0, mag_thin.shape[0] - 1):		   # rows
		for j in range(0, mag_thin.shape[1] - 1):	   # columns
			if mag_thin[i][j] >= tHigh:
				result_binary[i][j] = 1				 # set to 1 for >= tHigh
				
	return result_binary
	  
def Canny(img, sigma, tLow, tHigh):
	imgNorm = Normalize(img)
	Ix, Iy = DerivativesOfGaussian(imgNorm, 1)
	mag, orient = MagAndOrientation(Ix, Iy, tLow)
	mag_thin = NMS(mag, orient, tLow)
	result_binary = linking(mag_thin, orient, tLow, tHigh)
	return result_binary

#hought_transform - detecting lines

def hough_lines_acc(img, rho_res = 1, thetas = np.arange(-90, 90, 1)):
	rho_max = int(np.linalg.norm(img.shape-np.array([1,1]), 2));
	rhos = np.arange(-rho_max, rho_max, rho_res)
	thetas -= min(min(thetas),0)
	accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)
	yis, xis = np.nonzero(img) # use only edge points
	for idx in range(len(xis)):
		x = xis[idx]
		y = yis[idx]
		temp_rhos = x * np.cos(np.deg2rad(thetas)) + y * np.sin(np.deg2rad(thetas))
		temp_rhos = temp_rhos / rho_res + rho_max
		m, n = accumulator.shape
		valid_idxs = np.nonzero((temp_rhos < m) & (thetas < n))
		temp_rhos = temp_rhos[valid_idxs]
		temp_thetas = thetas[valid_idxs]
		c = np.stack([temp_rhos,temp_thetas], 1)
		cc = np.ascontiguousarray(c).view(np.dtype((np.void, c.dtype.itemsize * c.shape[1])))
		_,idxs,counts = np.unique(cc, return_index=True, return_counts=True)
		uc = c[idxs].astype(np.uint)
		accumulator[uc[:,0], uc[:,1]] += counts.astype(np.uint)
	accumulator = cv2.normalize(accumulator, accumulator, 0, 255,
								cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
	return accumulator, thetas, rhos 

def clip(idx):
	return int(max(idx,0))

def hough_peaks(H, numpeaks = 1, threshold = 150, nhood_size = 5):
	peaks = np.zeros((numpeaks,2), dtype=np.uint64)
	temp_H = H.copy()
	for i in range(numpeaks):
		_,max_val,_,max_loc = cv2.minMaxLoc(temp_H) # find maximum peak
		if max_val > threshold:
			peaks[i] = max_loc
			(c,r) = max_loc
			t = nhood_size//2.0
			temp_H[clip(r-t):int(r+t+1), clip(c-t):int(c+t+1)] = 0
		else:
			peaks = peaks[:i]
			break
	return peaks[:,::-1]
	
def hough_lines_draw(img, outfile, peaks, rhos, thetas):
    for peak in peaks:
        rho = rhos[peak[0]]
        theta = thetas[peak[1]] * np.pi / 180.0
        a = np.cos(theta); b = np.sin(theta)
        pt0 = rho * np.array([a,b])
        pt1 = tuple((pt0 + 1000 * np.array([-b,a])).astype(int))
        pt2 = tuple((pt0 - 1000 * np.array([-b,a])).astype(int))
        cv2.line(img, pt1, pt2, (0,255,0), 2)
    cv2.imwrite(outfile, img)
    return img

#corner detection
	
	
def line_eqution(pt1, pt2):
	x1, y1 = pt1 ; 	x2, y2 = pt2
	m = (y2 - y1)/(x2 - x1) if (x2 - x1) != 0 else (y2 - y1)/(0.1 ** 10)
	return (m, x1, y1)
	
def cross_point(line1, line2):
	m1, x1, y1 =  line1 
	m2, x2, y2 =  line2
	dm = (m1 - m2) if (m1 - m2) != 0 else 0.1 ** 10
	x = (-m2 * x2 + y2 + m1 * x1 - y1) / dm
	y = m1 * (x - x1) + y1
	ans = (int(round(x)), int(round(y)))
	return ans 

def findAllLines(peaks, rhos, thetas):
	lines = []
	for peak in peaks:
		rho = rhos[peak[0]]
		theta = thetas[peak[1]] * np.pi / 180.0
		a = np.cos(theta); b = np.sin(theta)
		pt0 = rho * np.array([a,b])
		pt1 = tuple((pt0 + 1000 * np.array([-b,a])).astype(int))
		pt2 = tuple((pt0 - 1000 * np.array([-b,a])).astype(int))
		line = line_eqution(pt1,pt2)
		lines.append(line)
	return lines

def findAllCP(lines, h, w):
	rng = range(len(lines))
	line_pairs = [[(lines[i], lines[j]) if i < j else 0 for i in rng] for j in rng]
	line_pairs = set(np.array(line_pairs).ravel()) - {0}
	CPs = [cross_point(i[0], i[1]) for i in line_pairs]
	CPs = [i if (i[0] in range(w)) & (i[1] in range(h)) else 0 for i in CPs]
	return set(CPs) - {0} 
 
def make_kernel(size):
	mid = int(size / 2)
	res = np.ones((size, size))
	res[mid:, :mid] *= -1
	res[:mid, mid:] *= -1
	res[:,mid] *= 0
	res[mid,:] *= 0
	return res.astype('int')

def corner_detector(pixels):
	if pixels.max() > 0:
		pixels = pixels / pixels.max()
	size = pixels.shape[0]
	kernel = make_kernel(size)
	return pixels.dot(kernel).trace()
	
def corner_xy(subimg, cp, ker_size = 7):
	res = np.zeros(subimg.shape)
	edge = int(len(subimg) / 2)
	ker_edge = int(ker_size / 2) + 2
	rng = range(ker_edge, len(subimg) - ker_edge)
	for i in range(len(subimg)):
		for j in range(len(subimg[0])):
			if (i in rng) & (j in rng):
				res[i, j] = abs(corner_detector(sub_img(subimg, (j, i), ker_size)))
	x0, y0 = np.subtract(cp, (edge,edge)) # the (0,0) location of the subimg
	y1, x1 = np.array(np.where(res == res.max()))[:,0] # max location in subimg
	x, y = x0 + x1, y0 + y1
	return [res.max(),(x, y)]

def to_order(points):
	def x(pt):
		return pt[0]
	def y(pt):
		return pt[1]
	by_order=[]
	by_y = sorted(points,reverse = False, key = y)
	upper = by_y[:2]
	lower = by_y[2:]
	upper_by_x = sorted(upper,reverse = False, key = x)
	lower_by_x = sorted(lower,reverse = False, key = x)
	by_order = [ upper_by_x[0] ,upper_by_x[1] ,lower_by_x[1] ,lower_by_x[0]] # up left, up right, down right, down left
	return by_order	

def findCorners(gray, peaks, rhos, thetas):
	img_cpy = gray.copy()
	lines = findAllLines(peaks, rhos, thetas)    
	h, w = gray.shape
	CPs = findAllCP(lines, h, w)
	subSize = 15
	edge = int(subSize / 2)
	ans = []
	chosenpoins = []
	for cp in CPs:           
		try:
			subimg = sub_img(gray, cp, subSize)
			pt = tuple(corner_xy(subimg, cp))
			chosenpoins.append(pt)
		except:
			continue
	
	
	chosenpoins = list(set(chosenpoins))
	close_groups = []
	for i in chosenpoins:
		group = [i]
		for j in chosenpoins:
			if (np.linalg.norm(np.subtract(i[1],j[1])) < 10):
				group.append(j)
		close_groups.append(group)
	close_groups = [tuple(set(i)) for i in close_groups]
	close_groups = list(set(close_groups))
	close_groups = [sorted(list(i), reverse = True) for i in close_groups]
	close_groups.sort(reverse = True)
	chosenpoins = [i[0][1] for i in close_groups][:4]
	chosenpoins = to_order(chosenpoins)
	
	return chosenpoins 

def Corners(img):

	gray_img = bgr_gray(img)
	smoothed_img = conv2D(gray_img, gauss_ker(5), 0)
	m, M, sig = 0.001, 0.5, 5
	edge_img = Canny(smoothed_img, sig, m, M)
	smoothed_edge_img = conv2D(edge_img, gauss_ker(0.5), 1)
	H, thetas, rhos = hough_lines_acc(smoothed_edge_img)
	peaks = hough_peaks(H, numpeaks = 6, threshold = 120, nhood_size = 40)
	corners = findCorners(gray_img, peaks, rhos, thetas)

	return corners


#warping	


def persp_trans(bg_img, img, p, old_loc):
	
	
	
	h, w = img.shape[:2]
	H, W = bg_img.shape[:2]
	chnls = img.shape[2] if img.ndim > 2 else 1
	#------------------------------ Ordering the points in p in the required order ------------------
	p = sorted(p, key = lambda x: np.linalg.norm(x))
	p[2], p[3] = p[3], p[2]
	p[1], p[3] = sorted([p[1], p[3]], key = lambda x: x[1])
	Xs, Ys = np.array(p).transpose()
	area = PolyArea(Xs, Ys)
	#------------------------------------ Creating the transpormation matrix ------------------------
	P = [[0, 0],[w - 1, 0],[w - 1, h - 1],[0, h - 1]]
	mat1 = []
	for i in range(4):
		mat1.append([-P[i][0], -P[i][1], -1, 0, 0, 0, P[i][0] * p[i][0], P[i][1] * p[i][0], p[i][0]])
		mat1.append([0, 0, 0, -P[i][0], -P[i][1], -1, P[i][0] * p[i][1], P[i][1] * p[i][1], p[i][1]])
	mat1.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
	transform = np.matrix(mat1).getI()[:,-1].reshape((3,3))
	D = np.linalg.det(transform)
	#---------------------------------- Calculating new coordinates for colors ----------------------
	wrpd_loc = np.array(transform.dot(np.array(old_loc).transpose()).transpose())
	cc = np.zeros((H, W)).astype('int').tolist()
	mask = np.zeros((H, W))
	for idx, I in enumerate(wrpd_loc):
		x, y = int(round(I[0] / I[2])), int(round(I[1] / I[2]))
		i, j = old_loc[idx][:2]
		if (x in range(W)) & (y in range(H)):
			mask[y,x] = 1
			if cc[y][x] == 0 : cc[y][x] = []
			cc[y][x].append(img[j, i])
	#------------------------------------------- Creating warped image ------------------------------
	cc = np.array(cc)
	color_loc = np.array(np.where(cc != 0)).transpose()
	zero_loc = np.array(np.where(cc == 0)).transpose()
	wrpd = np.zeros(bg_img.shape).astype('int')
	if chnls == 3:
		for i in color_loc:
			c_arr = np.array(cc[tuple(i)])
			wrpd[tuple(i)] = np.array([np.mean(c_arr[:,j]) for j in range(3)]).astype('int')
		for i in zero_loc:
			wrpd[tuple(i)] = np.array([0, 0, 0]).astype('int')
	else:
		for i in color_loc:
			wrpd[tuple(i)] = int(np.mean(np.array(cc[tuple(i)])))
	#------------------------------- Blending warped image into the background image -----------------
	mask = conv2D(mask.astype('float'), gauss_ker(10))
	mask[np.where(mask < 0.85)] = 0
	mask = conv2D(mask, gauss_ker(3))
	if chnls == 3:
		mask = gray_to_3ch(mask)
	res = (wrpd * (mask) + bg_img * (1 - mask)).astype('int')
	
	return (res, D, area)

#final video making 


def frames_dif(f1, f2):
	f1, f2 = bgr_gray(f1), bgr_gray(f2)
	dif = np.linalg.norm(f1 - f2)
	norm = np.prod(f1.shape) ** 0.5
	return dif / norm
	res = (int(dif / norm) * 2) - 5
	return res if res > 15 else 15


def vid_2_frames(vid, max_frames, start_at):
	frames = []
	count = 0
	vc = cv2.VideoCapture(vid)
	success, frame = vc.read()
	while success and count < max_frames :
		frames.append(frame)
		success, frame = vc.read()
		count += 1
	return frames[start_at:]
	  
def warpedlist(bg_vid, in_vid):
	start_at = 0
	max_frames = 300
	bg_frames = vid_2_frames(bg_vid, max_frames, start_at)
	in_frames = vid_2_frames(in_vid, max_frames, start_at)
	warpped_frames = []
	h, w = in_frames[0].shape[:2]
	old_loc = [[i % w, int((i - (i % w)) / w), 1] for i in range(h * w)]
	for i,frame in enumerate(bg_frames):
		t_start = time()
		if i == 0 :
			corners = np.array(Corners(bg_frames[0]))
			warped, D_prev, old_area = persp_trans(bg_frames[0], in_frames[0], corners, old_loc)
			warpped_frames.append(warped)
			#cv2.imwrite("frame%d.jpg" % (i + start_at), warped)
			print("frame%d.jpg" % (i + start_at))
		else:
			new_crnrs = []
			scores = []
			for j,cp in enumerate(corners):
				sub_size = 25
				edge = int(sub_size / 2)
				subimg = sub_img(bgr_gray(frame), cp, sub_size)
				corner_data = corner_xy(subimg, cp)
				scores.append(round(corner_data[0], 2))
				new_crnrs.append(corner_data[1])
			warped, D_curr, area = persp_trans(frame, in_frames[i], new_crnrs, old_loc)
			area_dif = abs(1 - (old_area / area))
			jerk = area_dif > 0.021
			lost_crnr = min(scores) < 2
			if jerk or lost_crnr:
				new_crnrs = np.array(Corners(frame))
				warped, d, area = persp_trans(frame, in_frames[i], new_crnrs, old_loc)
			#cv2.imwrite("frame%d.jpg" % (i + start_at), warped)
			print("frame%d.jpg" % (i + start_at))			
			warpped_frames.append(warped)
			corners = np.array(new_crnrs)
			D_prev = D_curr
			old_area = area
	return  warpped_frames
  
def makevideo (picturevideo,othervideo):

	warpedframes = warpedlist(picturevideo,othervideo)
	w,h,d = warpedframes[0].shape
	pathOut = 'finalvideo.avi'
	fps = 29.77
	size = (w,h) 
	writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc('M','J','P','G'), fps,(h,w))

	for frame in warpedframes:
		writer.write(frame.astype('uint8'))
	writer.release()


video1 = 'pictureOnTheWall.mp4'
video2 = 'strewberry.mp4'
makevideo(video1, video2)

