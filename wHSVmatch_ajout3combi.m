%% Dynamic feature computation
switch dataname
	case {'ETHZ1','ETHZ2','ETHZ3'}
		clear wHSVDyn
		ii = 1; ddtmp = 1;
		for i = 1:length(dset) % for each ped
			
			% dyn 1
			wHSVDyn(ii).hist = [];
			dd = uint16(length(ped(i).dynmodels)/2);
			rnddd = ped(i).rnddd; 
			for j = 1:dd % for each view/2
				ind = find(ped(i).dynmodels(rnddd(j)) == dset(i).globalindex); % we are searching for the local index (in permits_ind)
				wHSVDyn(ii).hist  =  [wHSVDyn(ii).hist,whisto2(:,dset(i).index(ind))];
			end
			ii = ii+1;
			
			% dyn 2
			wHSVDyn(ii).hist = [];
			for j = dd+1:length(ped(i).dynmodels) % for each view/2
				ind = find(ped(i).dynmodels(rnddd(j)) == dset(i).globalindex); % we are searching for the local index (in permits_ind)
				wHSVDyn(ii).hist  =  [wHSVDyn(ii).hist,whisto2(:,dset(i).index(ind))];
			end
			ii = ii+1;
			
		end
	case {'iLIDS','fablab','cbmi'}
		clear wHSVDyn
		ii = 1; ddtmp = 1;
		for i = 1:length(dset) % for each ped
			
			% dyn 1
			wHSVDyn(ii).hist = [];
			dd = uint16(length(ped(i).dynmodels)/2);
			rnddd = ped(i).rnddd; 
			for j = 1:dd % for each view/2
				ind = find(ped(i).dynmodels(rnddd(j)) == dset(i).globalindex); % we are searching for the local index (in permits_ind)
				wHSVDyn(ii).hist  =  [wHSVDyn(ii).hist,whisto2(:,dset(i).index(ind))];
			end
			ii = ii+1;
			
			% dyn 2
			wHSVDyn(ii).hist = [];
			for j = dd+1:length(ped(i).dynmodels) % for each view/2
				ind = find(ped(i).dynmodels(rnddd(j)) == dset(i).globalindex); % we are searching for the local index (in permits_ind)
				wHSVDyn(ii).hist  =  [wHSVDyn(ii).hist,whisto2(:,dset(i).index(ind))];
			end
			ii = ii+1;
			
        end
        case {'camnet'}
 		  clear wHSVDyn
        for i = 1:ncameras
            % Init
            wHSVDyn{i}.hist = [];
            Id_hist{i} = [];
            Beginning{i} = [];
            Ending{i} = [];
            
            index = false(1, length(ped)); 
            for k=1:length(ped)
                % We sort our data by cam
                index(k) = (ped(k).cam == cameras(i));
            end
            ped_selec{i} = ped(index);
            num_index = find(index == 1);
            
            % Init
            cpt = 1;
            
            for k = 1:length(ped_selec{i}) % for 1 cam
                wHSVDyn_tmp{cpt}.hist = [];
                
                for j = 1:nb_k_means
                    ind = find(ped(num_index(k)).dynmodels(j) == dset( ped(num_index(k)).id).globalindex); % we are searching for the local index (in permits_ind)
                    wHSVDyn_tmp{cpt}.hist = [wHSVDyn_tmp{cpt}.hist, whisto2(:, dset(ped(num_index(k)).id).index(ind))];
                    
                end
                cpt = cpt + 1;
                
                % Add id info
                Id_hist{i}= [Id_hist{i},  dset(ped(num_index(k)).id).id];
                % Add begin and end info
                Beginning{i} = [Beginning{i}, ped(num_index(k)).begin(1)];
                Ending{i} = [Ending{i}, ped(num_index(k)).end(1)];
                
            end
            wHSVDyn{i}.hist  =  [wHSVDyn{i}.hist,wHSVDyn_tmp];
           
            clear wHSVDyn_tmp ind index
        end
end
	
%% wHSV Hist distances computation 
hh = waitbar(0,'Distances computation wHSV...');

%Cameras combinaisons
CPairs = combnk(cameras,2);
numCPairs = size(CPairs, 1);

%k-means combinaisons
Combi_kmeans = perms(1:nb_k_means);

for n = 1:numCPairs
    distance = zeros(size(wHSVDyn{CPairs(n,2)}.hist,2),size(wHSVDyn{CPairs(n,1)}.hist,2));
    
    id_1 = Id_hist{CPairs(n,2)};
    id_2 = Id_hist{CPairs(n,1)};
    
    for i = 1:length(wHSVDyn{CPairs(n,2)}.hist)
        histtmp_i = wHSVDyn{CPairs(n,2)}.hist{i};
        
        for j = 1:length(wHSVDyn{CPairs(n,1)}.hist);
            histtmp_j = wHSVDyn{CPairs(n,1)}.hist{j};
            
            d = zeros(size(histtmp_i.hist,2),size(Combi_kmeans,1));
            for combi = 1:size(Combi_kmeans,1)
    %             if (i>=6)
    %                 figure(1)
    %                 plot(histtmp_i.hist);
    %                 figure(2)
    %                 plot(histtmp_j.hist);
    %             end
                
                for x = 1:nb_k_means
                    
                    % Bhattacharyya distance
                    d(x,combi) = bhattacharyya(histtmp_i.hist(:,x),histtmp_j.hist(:,Combi_kmeans(combi,x)));
                
                    %%% If we want to test another distance %%%
                    
%                     %X2 Distance
%                     H1 = histtmp_i.hist(:,x);
%                     H2 = histtmp_j.hist(:,Combi_kmeans(combi,x));
%                     for h = 1:length(histtmp_i.hist(:,x))
%                         if H1(h)+H2(h) ~= 0 
%                             d(x,combi) = d(x,combi) + (2 * (H1(h) - H2(h))^2)/(H1(h)+H2(h));
%                         end
%                     end
%                     %Jensen-Shannon Distance
%                     H1 = histtmp_i.hist(:,x);
%                     H2 = histtmp_j.hist(:,Combi_kmeans(combi,x));
%                     for h = 1:length(histtmp_i.hist(:,x))
%                         if H1(h)+H2(h) ~= 0 && H1(h) ~=0 && H2(h) ~= 0
%                             d(x,combi) = d(x,combi) + (H1(h)*log(2*(H1(h))/(H1(h)+H2(h)))+H2(h)*log(2*H2(h)/(H1(h)+H2(h))));
%                         end
%                     end

                end
                
            end
            
%             distance(i,j) = median(d(:)); % median of the distances
            
            [min_combi,indc_min] = min(d,[],2);
            [~,indl_min] = min(min_combi);
            distance(i,j) = min(d(:,indc_min(indl_min)),[],1); % min of the distances
        end
    end
    final_dist_hist{n} = distance;
    clear distance
    
   % waitbar(i+j+n/(size(wHSVDyn{1}.hist,2)+size(wHSVDyn{2}.hist,2)+size(wHSVDyn{3}.hist,2)),hh);
end
close(hh); 