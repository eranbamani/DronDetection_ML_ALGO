% Preprocessing
path = ('C:\Users\);
read_face = dir(path);
len_face = 440;%length(read_face);
pat = dir('C:\Users\');
a = pat(1).folder;
%
path2 = ('C:\Users\');
read_img = dir(path2);
len_img = 443;%length(read_img);
total = len_face + len_img;
pat2 = dir('C:\Users\');
b = pat2(1).folder;
% -------------------------------------------------------------------------
% data
width = 100;
len = 100;
label = [0 1 2 3];
img_size_max = 255;
img_exmp = imread(strcat(b,'\',read_img(1).name));
face_exmp = rgb2gray(imread(strcat(a,'\',read_face(1).name)));
%
% Read image
% face
for i = 1:len_face
    face = rgb2gray(imread(strcat(a,'\',read_face(i).name)));
    face = imresize(face,[width,len]);
    face = double(face/255);
    integralImage(:,:,i) = cumsum(cumsum(face,1),2);
end
% not_face
for j = 1:len_img
    notface = imread(strcat(b,'\',read_img(j).name));
    notface = imresize(notface,[width,len]);
    notface = double(notface/255);
    integralImage(:,:,j+i) = cumsum(cumsum(notface,1),2);
end
% -------------------------------------------------------------------------
%Adaboots Algorithm
[features,counter] = Adaboots(len_face,total,label,width,len,integralImage);

% Cascade
[finalerror] = cascade(total,label,counter,integralImage,features);

%--------------------------------------------------------------------------
% solution
err_f = num2str(finalerror/100);
dis = ['Mistake: ',err_f,'%'];
disp(dis);
img = '12.jpg';
img1 = '11.jpg';
img2 = 'image_0.jpg';
img3 = '0156.jpg';
Plot(img,img1,img2,img3,err_f,finalerror);