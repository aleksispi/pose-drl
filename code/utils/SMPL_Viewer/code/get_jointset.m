function [ind, labels] = get_jointset(varargin)
% get joint index based on the names of the subsets of joints

labels = {'Pelvis', 'RHip', 'RKnee','RAnkle','RToe','Site','LHip','LKnee','LAnkle','LeftToe','Site','Spine','Spine1','Neck','Head','Site','LShoulder','LShoulder','LElbow','LWrist','LThumb','Site','L_Wrist_End','Site','RShoulder','RShoulder','RElbow','RWrist','RThumb','Site','R_Wrist_End','Site'};
ind = [];
for i = 1: length(varargin)
  switch lower(varargin{i})
    case 'all'
      ind = 1:length(labels);
    case 'relevant'
      ind = [1 2 3 4 7 8 9 13 14 15 16 18 19 20 26 27 28];
    case 'relevant-2d'
      ind = [2 3 4 7 8 9 15 16 18 19 20 26 27 28];
    case 'body'
      ind = [ind 1 12 13];
    case 'body-all'
      ind = [ind 1 12 13 17 25];
    case 'torso'
      ind = [ind 1 2 7 14 18 26];
    case 'head'
      ind = [ind 14 15 16];
    case 'larm'
      ind = [ind 18 19 20];
    case 'larm-all'
      ind = [ind 18 19 20 21 22 23 24];
    case 'rarm'
      ind = [ind 26 27 28];
    case 'rarm-all'
      ind = [ind 26 27 28 29 30 31 32];  
    case 'root'
      ind = [ind 1];
    case 'lleg'
      ind = [ind 7 8 9];
    case 'lleg-all'
      ind = [ind 7 8 9 10 11];
    case 'rleg'
      ind = [ind 2 3 4];
    case 'rleg-all'
      ind = [ind 2 3 4 5 6];
    case 'lshoulder'
      ind = [ind 18];
    case 'rshoulder'
      ind = [ind 26];
    case 'lhip'
      ind = [ind 7];
    case 'rhip'
      ind = [ind 2];
    case 'rknee'
      ind = [ind 3];
    case 'rankle'
      ind = [ind 4];
    case 'lknee'
      ind = [ind 8];
    case 'lankle'
      ind = [ind 9];
    case 'lelbow'
      ind = [ind 19];
    case 'lwrist'
      ind = [ind 20];  
    case 'relbow'
      ind = [ind 27];
    case 'rwrist'
      ind = [ind 28];  
    otherwise
      error('Unknown joint!');
  end
end

% [ind,~,perm] = unique(ind);
% ind = ind(perm);
if nargout == 2
  labels = labels(ind);
end

end