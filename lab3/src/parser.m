addpath(genpath('./m'))
addpath(genpath('./data'))

format long g

data1 = csvread("Channel_1_900nm_2mm.csv")
data2 = csvread("Channel_2_900nm_2mm.csv")

data1(1,:) = []
data2(1,:) = []

data1_mv = data1(:,1)
data2_mv = data2(:,1)

data1_n = transpose(1:length(data1_mv))
data2_n = transpose(1:length(data2_mv))

data1_eps = 1e-4
data2_eps = 1e-4

data1_X = [ data1_n.^0 data1_n ];
data1_min_b = data1_mv - data1_eps
data1_max_b = data1_mv + data1_eps
[data1_coefs, data1_w] = minimization(data1_X, data1_min_b, data1_max_b)

data2_X = [ data2_n.^0 data2_n ];
data2_min_b = data2_mv - data2_eps
data2_max_b = data2_mv + data2_eps
[data2_coefs, data2_w] = minimization(data2_X, data2_min_b, data2_max_b)

fileID = fopen('data/Ch1.txt','w');
fprintf(fileID,'%g %g\n', data1_coefs(1), data1_coefs(2));
for c = 1 : length(data1_w)
  fprintf(fileID, "%g\n", data1_w(c));
end
fclose(fileID);

fileID = fopen('data/Ch2.txt','w');
fprintf(fileID,'%g %g\n', data2_coefs(1), data2_coefs(2));
for c = 1 : length(data2_w)
  fprintf(fileID, "%g\n", data2_w(c));
end
fclose(fileID);