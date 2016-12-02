import pyedflib
import pyeeg
import numpy as np
import math
from scipy import signal 
import obspy
from obspy.imaging.cm import obspy_sequential
from obspy.signal.tf_misfit import cwt

f = pyedflib.EdfReader("Downloads/a_1.edf")
print("\nlibrary version: %s" % pyedflib.version.version)

print("\ngeneral header:\n")

# print("filetype: %i\n"%hdr.filetype);
print("edfsignals: %i" % f.signals_in_file)
print("file duration: %i seconds" % f.file_duration)
print("startdate: %i-%i-%i" % (f.getStartdatetime().day,f.getStartdatetime().month,f.getStartdatetime().year))
print("starttime: %i:%02i:%02i" % (f.getStartdatetime().hour,f.getStartdatetime().minute,f.getStartdatetime().second))
# print("patient: %s" % f.getP);
# print("recording: %s" % f.getPatientAdditional())
print("patientcode: %s" % f.getPatientCode())
print("gender: %s" % f.getGender())
print("birthdate: %s" % f.getBirthdate())
print("patient_name: %s" % f.getPatientName())
print("patient_additional: %s" % f.getPatientAdditional())
print("admincode: %s" % f.getAdmincode())
print("technician: %s" % f.getTechnician())
print("equipment: %s" % f.getEquipment())
print("recording_additional: %s" % f.getRecordingAdditional())
print("datarecord duration: %f seconds" % f.getFileDuration())
print("number of datarecords in the file: %i" % f.datarecords_in_file)
print("number of annotations in the file: %i" % f.annotations_in_file)

channel = 3
print("\nsignal parameters for the %d.channel:\n\n" % channel)

print("label: %s" % f.getLabel(channel))
print("samples in file: %i" % f.getNSamples()[channel])
# print("samples in datarecord: %i" % f.get
print("physical maximum: %f" % f.getPhysicalMaximum(channel))
print("physical minimum: %f" % f.getPhysicalMinimum(channel))
print("digital maximum: %i" % f.getDigitalMaximum(channel))
print("digital minimum: %i" % f.getDigitalMinimum(channel))
print("physical dimension: %s" % f.getPhysicalDimension(channel))
print("prefilter: %s" % f.getPrefilter(channel))
print("transducer: %s" % f.getTransducer(channel))
print("samplefrequency: %f" % f.getSampleFrequency(channel))
#annotations = f.readAnnotations()
#for n in np.arange(f.annotations_in_file):
 #   print("annotation: onset is %f    duration is %s    description is %s" % (annotations[0][n],annotations[1][n],annotations[2][n]))

buf = f.readSignal(channel)
n = 10000
print("\nread %i samples\n" % n)
result = ""
for i in np.arange(n):
    result += ("%.1f, " % buf[i])
#print(result)



def parse_signal(f):

	N = 1250
	num_sec = 0  

	print f.datarecords_in_file

	while num_sec < f.file_duration: 

		print"here"
		channel_window_data = []

		num_signals = f.signals_in_file 
		for i in range(1,num_signals+1):
			channel = i 
			buf = f.readSignal(i)
			buf_window = [] 
			for j in np.arange(N):
				buf_window.append(buf[j+(N-1250) ])
			channel_window_data.append(buf_window)
			calc_features(channel_window_data)
	num_sec += 5 
	N += 1250 

def calc_features(channel_window_data):
	for i in range(0,len(channel_window_data)):
		for j in range(0,len(channel_window_data)):
			if i != j:
				cross_correlation = calc_cross_correlation(channel_window_data[i],channel_window_data[j])
				non_linear_interdependence = calc_non_linear_interdependence(channel_window_data[i],channel_window_data[j])
				stl_exponent = calc_stl_exponent(channel_window_data[i],channel_window_data[j])
				synchrony = calc_synchrony(channel_window_data[i],channel_window_data[j])
				coherence = calc_coherence(channel_window_data[i],channel_window_data[j])


def calc_cross_correlation(x,y):

	N = 1250 
	x_x = np.correlate(x,x,"valid")
	y_y = np.correlate(y,y,"valid")
	z = np.correlate(x,y,"full")
	z_subset = [] 
	num_overlap = 1 
	for i in range(0,len(z)):
		if num_overlap >= 1125:
			z_subset.append(z[i])
		if i < 1250: 
			num_overlap += 1 
		else:
			num_overlap -= 1 
	denom = np.sqrt(x_x*y_y)
	max_val = max(z_subset)
	return max_val /denom 

def calc_non_linear_interdependence(x,y):

	print(len(x))
	x_embedded = pyeeg.embed_seq(np.array(x),6,10)
	y_embedded = pyeeg.embed_seq(np.array(y),6,10)

	interdependence_x_y = calc_non_linear_interdependence_helper(x_embedded,y_embedded)
	interdependence_y_x = calc_non_linear_interdependence_helper(y_embedded,x_embedded)

	return (interdependence_y_x + interdependence_x_y)/2

def calc_non_linear_interdependence_helper(x,y):

	K = 5

	ret = 0.0 


	num_rows = np.shape(x)[0]

	for i in range(0,num_rows):
		x_val = x[i][0] #first element in embedded matrix 
		y_val = y[i][0]
		nn_time_x = get_nn(x[i][1:],x_val,K) #get K nn time indices 
		nn_time_y = get_nn(y[i][1:],y_val,K)
		
		total_x = 0.0
		total_x_given_y = 0.0 
		for j in range(0,K):
			x_index = nn_time_x[j]
			y_index = nn_time_y[j]
			total_x += np.linalg.norm(x_val-x[i][x_index])
			total_x_given_y += np.linalg.norm(x_val-x[i][y_index])


		total_x = total_x/K
		total_x_given_y = total_x_given_y/K

		ret += total_x/total_x_given_y

	return ret/num_rows 


def calc_stl_exponent(x,y):

	x_embedded = pyeeg.embed_seq(np.array(x),6,7)
	y_embedded = pyeeg.embed_seq(np.array(y),6,7)

	num_rows = np.shape(x_embedded)[0]

	x_total = 0.0
	y_total  = 0.0

	for i in range(0,num_rows):
		x_curr = x_embedded[i]
		y_curr = y_embedded[i]

		x_val = x_embedded[i][0]
		y_val = y_embedded[i][0]

		x_total +=  math.log(abs(x_embedded[i][2]/x_val),2)
		y_total += math.log(abs(y_embedded[i][2]/y_val),2)

	x_total = x_total/(num_rows*12)
	y_total = y_total/(num_rows*12)


def calc_synchrony(x,y):

	w_t_x = cwt(x, .004, 8, 4, 100,7,'morlet') #not sure about 3rd parameter 
	w_t_y = cwt(y, .004, 8, 4, 100,7,'morlet') #not sure about 3rd parameter 

	result = 0.0 

	num_rows = np.shape(w_t_x)[0]


	ret = [] 

	for f in range(0,num_rows): 
		coef_w_x = np.dot(x,w_t_x[f])
		coef_w_y = np.dot(y,w_t_y[f])
		coef_w_y_c = np.dot(y,w_t_y[f]).conjugate() 
		val = coef_w_x*coef_w_y_c/(abs(coef_w_x)*abs(coef_w_y))
		ret.append(val)

	return ret #val for every frequency 




def calc_coherence(x,y): 

	return signal.coherence(x, y)

def get_nn(x,val,K):

	x_c = np.array(x) 
	nn = []
	for i in range(0,K):
		nearest_index = find_nearest(x_c,val) 
		nn.append(nearest_index)
		x_c = np.delete(x_c,nearest_index)
	return nn 

def find_nearest(array,value):


	idx = (np.abs(array-value)).argmin()
	return idx 


def parse_annotation_file(f):

	


parse_signal(f)