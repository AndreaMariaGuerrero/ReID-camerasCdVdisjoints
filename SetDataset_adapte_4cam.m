function [dset, idp] = SetDataset_adapte(permit_inds,dir_e_list,all_pic_nb,real_nb_frames,all_begin,all_end)
% dset = SetDataset_adapte(permit_inds,dir_e_list,all_pic_nb,real_nb_frames)
%
% extract the index of each view of a single pedestrian
% dset is indexed by the pedestrian ID and contains the relative and global 
% indexes of the different views (imgs into the dataset)
%
% all_pic_nb : give the picture order that k-means gave
% real_nb_frames : give the real frame number extracted in every video
% all_begin : give the first frame of the tracklet corresponding to all_pic_nb
% all_end : give the last frame of the tracklet corresponding to all_pic_nb

j = 0; idpersonold = 0;

min_frames = min(real_nb_frames);

for i=1:length(permit_inds)
    nb_im = all_pic_nb(i);
    n = dir_e_list(nb_im).name;

    Liste = {dir_e_list(:).name} ; % transform a mat into cell
    %Search for the right pic (corresponding to nb_im pic number)
    for li = 1:length(Liste)
        matches = regexp( Liste{li},'\d+','match'); % \d+ : search all number
        if (str2num(matches{2}) == nb_im)
            break;
        end
    end
    
%     img = picture{nb_im};
%     imshow(img); 
    
    %extract id
    idp = str2num(matches{1});
    %extract detection number
    npic = str2num(matches{2});
    %extract frame number
    nimage = str2num(matches{3});
    %extract camera number
    ncam = str2num(matches{4});
    
    if (idp ~= idpersonold)    
        j = j+1; % It's a new person
        dset(j).globalindex = permit_inds(i);
        dset(j).index = i;
        dset(j).cam = ncam;
        dset(j).frame = real_nb_frames(nimage)-min_frames+1;
        dset(j).pic = npic;
        dset(j).begin = all_begin(i);
        dset(j).end = all_end(i);
        idpersonold = idp;
    else
        dset(j).globalindex = [dset(j).globalindex permit_inds(i)];
        dset(j).index = [dset(j).index i];
        dset(j).cam = [dset(j).cam ncam];
        dset(j).frame = [dset(j).frame nimage];
        dset(j).pic = [dset(j).pic npic];
        dset(j).begin = [dset(j).begin all_begin(i)];
        dset(j).end = [dset(j).end all_end(i)];
    end
    dset(j).id = idp;

    
    
end

