import string

print "Processing results"

f = open('./results2.txt','r')

line = f.readline()

num = float(string.rsplit(line).pop())

print num, 'entries total'

#strings = string.split(f.readline())

#avg_gpu_dens = float(strings.pop(0));
#avg_cpu_dens = float(strings.pop(0));
#avg_cpu_temp = float(strings.pop(0));
#avg_gpu_temp = float(strings.pop(0));
#idx = 1;

avg_gpu_dens = 0.0;
avg_cpu_dens = 0.0;
avg_cpu_temp = 0.0;
avg_gpu_temp = 0.0;

for line in f:
	strings = string.split(line)
#	idx = idx + 1	
	avg_gpu_dens = avg_gpu_dens + float(strings.pop(0))/num;
	avg_cpu_dens = avg_cpu_dens + float(strings.pop(0))/num;
	avg_cpu_temp = avg_cpu_temp + float(strings.pop(0))/num;
	avg_gpu_temp = avg_gpu_temp + float(strings.pop(0))/num;

#	avg_gpu_dens = avg_gpu_dens*(idx-1)/idx + float(strings.pop(0))/idx;
#	avg_cpu_dens = avg_cpu_dens*(idx-1)/idx + float(strings.pop(0))/idx;
#	avg_cpu_temp = avg_cpu_temp*(idx-1)/idx + float(strings.pop(0))/idx;
#	avg_gpu_temp = avg_gpu_temp*(idx-1)/idx + float(strings.pop(0))/idx;

print "Avg GPU Dens || Avg CPU Dens || Avg GPU Temp || Avg CPU Temp"
print avg_gpu_dens,  avg_cpu_dens,  avg_gpu_temp,  avg_cpu_temp
print "Density ratio (GPU/CPU): " , avg_gpu_dens/avg_cpu_dens , "Temp ratio (GPU/CPU): " , avg_gpu_temp/avg_cpu_temp 

f.close();
