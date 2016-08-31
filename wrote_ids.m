%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             author : Guerrero Andréa, stagiaire LAAS/CNRS               %
%             subject : give ids to test re-ID efficiency                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


hwait = waitbar(0,'build imagette and mask...');

% count nb of detections for every frame
[ubbs, ind] = unique(bbs(:,8));
nb_per_frame = histc(bbs(:,8),ubbs);
% same for ground truth
ubbs_gt = unique(bbs_gt(:,1));
nb_per_frame_gt = histc(bbs_gt(:,1),ubbs_gt);

 figure;
cpt=1; cpt_gt=1; l=1; seuil = 0.7; % overlap accepted         %seuil = 20; %error tolerance : 'seuil' pixels
ind_current = ubbs(cpt);
while( ind_current ~= ubbs(end) )
    x = find(bbs(:,8) == ind_current);
    for i = 1:length(x)
        n = x(i);
        im_r = list_images{bbs(n,1)};
        im_resize = imresize(im_r, 2,'bilinear');
        % Num cam
        cam_current = bbs(n,7);
        % Bounds of bbs
        borne_lt = round(bbs(n,3));
        if(borne_lt<1), borne_lt = 1;  end;
        borne_ld = round(bbs(n,5));
        borne_l = borne_lt + borne_ld;
        if(borne_l>size(im_resize,1)), borne_l =  size(im_resize,1);  end;
        borne_rt = round(bbs(n,2));
        if(borne_rt<1), borne_rt = 1;  end;
        borne_rd = round(bbs(n,4));
        borne_r = borne_rt + borne_rd;
        if(borne_r>size(im_resize,2)), borne_r =  size(im_resize,2);  end;
        
        % center of bbs
        x_center = (borne_lt + borne_l)/2;
        y_center = (borne_rt + borne_r)/2;

%         imshow(im_resize);hold on;
%         plot(y_center,x_center, 'xr');
%         pause(0.25);
        
        y = find(bbs_gt(:,1) == ind_current );
        if(~isempty(y))
            for j = 1:length(y)
                n_gt= y(j);
                if(bbs_gt(n_gt,7) == cam_current)
                    % recovery area
                    overlapRatio = bboxOverlapRatio(bbs(n,2:5), bbs_gt(n_gt,2:5)*2);

                       % Center of bbs
%                        x_center_gt = (2*bbs_gt(n_gt,3) + bbs_gt(n_gt,5)); % Warning : Not same size !
%                        y_center_gt = (2*bbs_gt(n_gt,2) + bbs_gt(n_gt,4));


                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % -- Build 'imagette' needed to apply SDALF -- %
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    %if(abs(x_center-x_center_gt) < seuil && abs(y_center-y_center_gt) < seuil)
                    if (overlapRatio > seuil )
                        
                        %Stock info bbs
                        stock_xcenter{l} = x_center;
                        stock_ycenter{l} = y_center;
                        stock_bbs{l} = bbs(n,2:8);

                        %Build imagette
                        imagette_gt = im_r(bbs_gt(n_gt,3):bbs_gt(n_gt,3)+bbs_gt(n_gt,5),bbs_gt(n_gt,2):bbs_gt(n_gt,2)+bbs_gt(n_gt,4), :);
                        imagette = im_resize(borne_lt:borne_l, borne_rt:borne_r ,:);
                        picture{l}     =   imresize(imagette,[H,W], 'bilinear' ); % normalization 64x128
                        imwrite(picture{l}, [rep_imagette 'personne' num2str(bbs_gt(n_gt,6),'%2.2d'), 'detection' num2str(l,'%4.4d')  'frame', num2str(bbs(n,1), '%3.3d'), 'cam', num2str(bbs(n,7), '%d'), '.jpg']);
                        picture_gt{l} = imresize(imagette_gt,[H,W], 'bilinear' );
                        
% %                         subplot(1,2,1), imshow(picture{l});
% %                         subplot(1,2,2), imshow(picture_gt{l});title('Paper response');
% %                         pause(0.25);
            
                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                % -- Build 'mask' needed to background substraction -- %
                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        
                        if maskon
                            m_curr = msk{1,bbs(n,1)};% keep mask of current picture
                            m_curr_r = imresize(m_curr, 2,'bilinear');
                            m_c = m_curr_r(borne_lt:borne_l, borne_rt:borne_r);
                            mask_fin(:,:,l) = imfill(imresize(m_c,[H,W],'bilinear'),'holes');
    %                                 subplot(1,2,1), imshow(picture{l} );hold on;
    %                                 rectangle('Position', [borne_rt borne_lt borne_rd borne_ld], 'EdgeColor','r')
    %                                 subplot(1,2,2), imshow(mask_fin(:,:,l));
                        else
                            mask_fin(:,:,l) = ones(H,W);
                        end
                        l = l+1;
                    end
                end
                
            end
        end

        clear m_curr m_curr_r m_c;
        
        
    end
    
    
    
    cpt = cpt+1; %num_frame_current = nb_frames(cpt);
    ind_current = ubbs(cpt);
    waitbar(cpt/size(ubbs,1),hwait)
end
close(hwait)

num_frames = length(picture);

fprintf('Imagette OK \n');
%%
% figure;
% for i = 1:length(test_num)
%         im_r = list_images{test_num{i}};
%         im_resize = imresize(im_r, 2,'bilinear');
%         imshow(im_resize);hold on;
%         plot(stock_ycenter{i}, stock_xcenter{i},'xr');
%         hold off;
%         pause(0.25);
% 
% end


