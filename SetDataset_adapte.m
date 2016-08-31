function [dset, idp] = SetDataset_adapte(permit_inds,dir_e_list,all_pic_nb)
% dset = SetDataset(permit_inds,dir_e_list)
%
% extract the index of each view of a single pedestrian
% dset is indexed by the pedestrian ID and contains the relative and global 
% indexes of the different views (imgs into the dataset)

j = 0; idpersonold = 0;

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
        dset(j).frame = nimage;
        dset(j).pic = npic;
        idpersonold = idp;
    else
        dset(j).globalindex = [dset(j).globalindex permit_inds(i)];
        dset(j).index = [dset(j).index i];
        dset(j).cam = [dset(j).cam ncam];
        dset(j).frame = [dset(j).frame nimage];
        dset(j).pic = [dset(j).pic npic];
    end
    dset(j).id = idp;

    
    
end

