function ksvddenoisedemo
%KSVDDENOISEDEMO K-SVD denoising demonstration.
%  KSVDDENISEDEMO reads an image, adds random white noise and denoises it
%  using K-SVD denoising. The input and output PSNR are compared, and the
%  trained dictionary is displayed.
%
%  To run the demo, type KSVDDENISEDEMO from the Matlab prompt.
%
%  See also KSVDDEMO.


%  Ron Rubinstein
%  Computer Science Department
%  Technion, Haifa 32000 Israel
%  ronrubin@cs
%
%  August 2009


disp(' ');
disp('  **********  K-SVD Denoising Demo  **********');
disp(' ');
disp('  This demo reads an image, adds random Gaussian noise, and denoises the image');
disp('  using K-SVD denoising. The function displays the original, noisy, and denoised');
disp('  images, and shows the resulting trained dictionary.');
disp(' ');


%% prompt user for image %%

pathstr = fileparts(which('ksvddenoisedemo')); %which查找文件路径，fileparts获取文件名的组成部分
dirname = fullfile(pathstr, 'images', '*.png'); %fullfile 函数作用是作用是利用文件各部分信息创建并合并成完整文件名
imglist = dir(dirname); %dir函数是最常用的转换路径的函数，可以获得指定文件夹下的所有子文件夹和文件

disp('  Available test images:');
disp(' ');
for k = 1:length(imglist)
  printf('  %d. %s', k, imglist(k).name);
end
disp(' ');

%这个函数让你选择一张图片进行去噪
imnum = 0;
while (~isnumeric(imnum) || ~iswhole(imnum) || imnum<1 || imnum>length(imglist))
  imnum = input(sprintf('  Image to denoise (%d-%d): ', 1, length(imglist)), 's');
  imnum = sscanf(imnum, '%d');
end

imgname = fullfile(pathstr, 'images', imglist(imnum).name);



%% generate noisy image %%
%这个函数产生噪声图片，就是产生一个同大小的随机矩阵*sigma，把这个噪声矩阵加到原图上去
sigma = 20;

disp(' ');
disp('Generating noisy image...');

im = imread(imgname);
im = double(im);

n = randn(size(im)) * sigma;
imnoise = im + n;



%% set parameters %%
%这个参数的设置是来自于ksvddenoise函数里所需要的

params.x = imnoise;
params.blocksize = 8;
params.dictsize = 256;
params.sigma = sigma;
params.maxval = 255;
params.trainnum = 40000;
params.iternum = 20;
params.memusage = 'high';



% denoise!
%用ksvddenoise函数来对上面的图片进行去噪
disp('Performing K-SVD denoising...');
[imout, dict] = ksvddenoise(params);



% show results %
%PSNR的值代表的是峰值信噪比
dictimg = showdict(dict,[1 1]*params.blocksize,round(sqrt(params.dictsize)),round(sqrt(params.dictsize)),'lines','highcontrast');
figure; imshow(imresize(dictimg,2,'nearest'));
title('Trained dictionary');

figure; imshow(im/params.maxval);
title('Original image');

figure; imshow(imnoise/params.maxval); 
title(sprintf('Noisy image, PSNR = %.2fdB', 20*log10(params.maxval * sqrt(numel(im)) / norm(im(:)-imnoise(:))) ));

figure; imshow(imout/params.maxval);
title(sprintf('Denoised image, PSNR: %.2fdB', 20*log10(params.maxval * sqrt(numel(im)) / norm(im(:)-imout(:))) ));

