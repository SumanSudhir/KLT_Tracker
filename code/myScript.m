clear;
close all;
warning('off','all')
clc;

%% Your Code Here
I = imread("../input/1.jpg");
%figure(1),imshow(I);

%Corner Detection
C = corner(I);
imshow(I)

hold on
%plot(C(:,1),C(:,2),'r*');
plot(180,259,'r*');
plot(155,244,'g*');

%Point to consider is (155,244) and (180,259) from the corner image
%points = cell(1,2);
points{1} = [244 155];
points{2} = [259 180 ];
%-------------------------------------------------------
%%%%% KLT Tracker
%%
[rows, colms] = size(I);
window_size = 22;

for i = 1:2
  if (points{i}(1)-window_size > 0 && points{i}(1)+window_size <= rows && points{i}(2)-window_size > 0 && points{i}(2)+window_size <= colms )
    
    p = [0 0 0 0 points{i}(1) points{i}(2)];
    
    count = 0;
    n_count = 1;
    %Template image
    T = I(points{i}(1)-window_size:points{i}(1)+window_size,points{i}(2)-window_size:points{i}(2)+window_size);

    %figure , imshow(T)
    T = double(T);
    %Rectangular grid in 2-D space

    [x,y]=ndgrid(-(size(T,1)-1)/2:1:(size(T,1)-1)/2,-(size(T,2)-1)/2:1:(size(T,2)-1)/2);

    %loop for all the the given frames

    for j = 2:247
      if j ~= 60
      next_frame = imread(strcat('../input/',num2str(j),'.jpg'));
      end

      frame_copy = next_frame;
      copy_p = p;
      next_frame = double(next_frame);
      sigma = 1.4;
      delta_p = 5;

      %For taking the derivative of image
      
       [x_range,y_range]=ndgrid(floor(-3*sigma):ceil(3*sigma),floor(-3*sigma):ceil(3*sigma));
       
       dx=-(x_range./(2*pi*sigma^4)).*exp(-(x_range.^2+y_range.^2)/(2*sigma^2));
       
       dy=-(y_range./(2*pi*sigma^4)).*exp(-(x_range.^2+y_range.^2)/(2*sigma^2));
       
       % Filter the images to get the derivatives
       I_gradient_x = imfilter(next_frame,dx,'conv');
       I_gradient_y = imfilter(next_frame,dy,'conv');
       
      %imshow(I_gradient_y)

      counter = 0;
      threshold = 0.02;

      while (norm(delta_p)>threshold)
        counter = counter + 1;
        %Break if not converges for 60 loop
        if (counter > 60)
          break;
        end

        %Affine matrix for rotation and translation
        W_p = [1+p(1) p(3) p(5); p(2) 1+p(4) p(6)];

        %************************As per the provided paper the KLT tracker is divided into 9 steps as follows *********************************

        %Step 1
        %Warp the image with W
        I_warp = warpping(next_frame,x,y,W_p);

        %Step 2
        I_error = T - I_warp;

        %Break if windcow goes outside image
        if ((p(5)>(size(next_frame,1))-1)||(p(6)>(size(next_frame,2)-1))||(p(5)<0)||(p(6)<0))
           break;
        end

        %Step 3: Warp the gradient
        I_x = warpping(I_gradient_x,x,y,W_p);
        I_y = warpping(I_gradient_y,x,y,W_p);

        %Step 4: Computation of the Jacobian
        Jacob_x = [x(:) zeros(size(x(:))) y(:) zeros(size(x(:))) ones(size(x(:))) zeros(size(x(:)))];
        Jacob_y = [zeros(size(x(:))) x(:) zeros(size(x(:))) y(:) zeros(size(x(:))) ones(size(x(:)))];

        %Step 5: Computation of steepest descent
        I_steep = zeros(numel(x),6);
        for l = 1:numel(x)
          W_jacob = [Jacob_x(l,:);Jacob_y(l,:)];
          Grad = [I_x(l) I_y(l)];
          I_steep(l,1:6) = Grad*W_jacob;
        end

        %Step 6: Computation of Hessian
        H = zeros(6,6);
        for m = 1:numel(x)
          H = H + I_steep(m,:)'*I_steep(m,:);
        end

        %Step: 7 Multiplication of steepest descend with Error
        summation = zeros(6,1);
        for n=1:numel(x)
           summation = summation + I_steep(n,:)'*I_error(n);
        end

        %Step: 8 Computer delta_p
        delta_p = H\summation;    
        %delta_p = inv(summation)*H;

        %Step 9 Update the parameters delat_p
         p = p + delta_p' ;

      end

      count = count + 1;
      if (count == 20)
       
       T = frame_copy(p(5)-window_size:p(5)+window_size,p(6)-window_size:p(6)+window_size);
       p = [0 0 0 0 p(5) p(6)];
       %Convert to double
       T= double(T);
       [x,y]=ndgrid(-(size(T,1)-1)/2:1:(size(T,1)-1)/2,-(size(T,2)-1)/2:1:(size(T,2)-1)/2);
       count = 0;
      end

      %In order to draw mark on image store corners
      %p      
      if(i==1)
        n_corners1{n_count} = [p(6) p(5)];
      end

      if(i==2)
        n_corners2{n_count} = [p(6) p(5)];
      end
      n_count = n_count + 1;

    end
  end
end

n_count = 1;
%%
%Matlab was getting crashed when I was trying to save more than 200 images
%therefore I saved from (61 to 247) 

%% Save all the trajectories frame by frame
% variable trackedPoints assumes that you have an array of size 
% No of frames * 2(x, y) * No Of Features
% noOfFeatures is the number of features you are tracking
% Frames is array of all the frames(assumes grayscale)
noOfPoints = 1;
for i=61:247
    NextFrame = imread(strcat('../input/',num2str(i),'.jpg'));
    imshow(uint8(NextFrame)); 
    hold on;
    
    for nF = 1:noOfPoints
      plot(n_corners1{59 + nF}(1),n_corners1{59 + nF}(2),'g*');
      plot(n_corners2{59 + nF}(1),n_corners2{59 + nF}(2),'b*');
    end
    hold off;
    
    saveas(gcf,strcat('../output/',num2str(i),'.jpg'));
    close all;
    noOfPoints = noOfPoints + 1;
end 
   


%Warping
%%
function I_warpped = warpping(I_input,x,y,M)
    % Affine transformation function (Rotation, Translation, Resize)

    % Calculate the Transformed coordinates
    local_x =  M(1,1) * x + M(1,2) *y + M(1,3);
    local_y =  M(2,1) * x + M(2,2) *y + M(2,3);

    % All the neighborh pixels involved in linear interpolation.
    x_b0=floor(local_x);
    y_b0=floor(local_y);
    x_b1=x_b0+1;
    y_b1=y_b0+1;

    % Linear interpolation constants (factor)
    x_m=local_x-x_b0;
    y_m=local_y-y_b0;
    factor_0=(1-x_m).*(1-y_m);
    factor_1=(1-x_m).*y_m;
    factor_2=x_m.*(1-y_m);
    factor_3=x_m.*y_m;


    check_x_b0=(x_b0<0)|(x_b0>(size(I_input,1)-1));
    check_y_b0=(y_b0<0)|(y_b0>(size(I_input,2)-1));
    x_b0(check_x_b0)=0;
    y_b0(check_y_b0)=0;
    check_x_b1=(x_b1<0)|(x_b1>(size(I_input,1)-1));
    check_y_b1=(y_b1<0)|(y_b1>(size(I_input,2)-1));
    x_b1(check_x_b1)=0;
    y_b1(check_y_b1)=0;

    I_warpped=zeros([size(x) size(I_input,3)]);
    for i=1:size(I_input,3)
        I_input_one=I_input(:,:,i);

        intensity_xyz0=I_input_one(1+x_b0+y_b0*size(I_input,1));
        intensity_xyz1=I_input_one(1+x_b0+y_b1*size(I_input,1));
        intensity_xyz2=I_input_one(1+x_b1+y_b0*size(I_input,1));
        intensity_xyz3=I_input_one(1+x_b1+y_b1*size(I_input,1));
        I_warpped_one=intensity_xyz0.*factor_0+intensity_xyz1.*factor_1+intensity_xyz2.*factor_2+intensity_xyz3.*factor_3;
        I_warpped(:,:,i)=reshape(I_warpped_one, [size(x,1) size(x,2)]);
    end
end
%%
