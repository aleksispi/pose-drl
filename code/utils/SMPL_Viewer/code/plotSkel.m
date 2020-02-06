function [ output_args ] = plotSkel( pose3D, colorOption )

if (exist('/cluster/work/', 'dir'))
    workPath = '/cluster/work/alin/';
else
    workPath = '/work/scratch/users/alin/';
end;

if(nargin == 0)
    d = load('predictions/pose3D_E9.mat');
    pose = squeeze(d.pose3D(400, :, :));
    %     i = 1000;
    %     poseGT = load([workPath '/DeepHumanReconstruction_Data/h36m_4000_data/Test/poseData.mat']);
    [ind, names] = get_jointset('relevant');
    pose3D = zeros(32, 3);
    pose3D(ind, :) = pose;
    %     pose3D_GT = poseGT.data3D_Test;
    %     pose2D_GT = pose
    %     gtruth = pose3D_GT(i, :);
    %     gtruth = reshape(gtruth, 3, 32)';
    % %     gtruth = gtruth - repmat(gtruth(1, : ), 32, 1);
    % %    gtruth = gtruth(ind, :);
    %     pose3D = gtruth';
    pose3D = pose3D';
    colorOption = 'r';
end

%PLOTSKEL Summary of this function goes here
%   Detailed explanation goes here
%
if (size(pose3D, 1) ~= 32 && size(pose3D, 2) == 3)
    buff_large = zeros(32, 3);
    [ind, ~] = get_jointset('relevant');
    buff_large(ind, :) = pose3D;
    pose3D = buff_large';
end;
%%%
% img = zeros(1002, 1000, 3);

% plot3(pose3D(order(1), :), pose3D(order(2), :), pose3D(order(3), :), [colorOption '*']);
% figure;
hold on;
if(size(pose3D, 1) == 3)
    order = [1 3 2];
    plot3(pose3D(order(1), [1 13]), pose3D(order(2), [1 13]), -pose3D(order(3), [1 13]), [colorOption '-'], 'LineWidth',5);
    plot3(pose3D(order(1), [13 14]), pose3D(order(2), [13 14]), -pose3D(order(3), [13 14]), [colorOption '-'], 'LineWidth',5);
    plot3(pose3D(order(1), [14 15]), pose3D(order(2), [14 15]), -pose3D(order(3), [14 15]), [colorOption '-'], 'LineWidth',5);
    plot3(pose3D(order(1), [16 15]), pose3D(order(2), [16 15]), -pose3D(order(3), [16 15]), [colorOption '-'], 'LineWidth',5);
    plot3(pose3D(order(1), [14 18]), pose3D(order(2), [14 18]), -pose3D(order(3), [14 18]), [colorOption '-'], 'LineWidth',5);
    plot3(pose3D(order(1), [18 19]), pose3D(order(2), [18 19]), -pose3D(order(3), [18 19]), [colorOption '-'], 'LineWidth',5);
    plot3(pose3D(order(1), [19 20]), pose3D(order(2), [19 20]), -pose3D(order(3), [19 20]), [colorOption '-'], 'LineWidth',5);
    plot3(pose3D(order(1), [14 26]), pose3D(order(2), [14 26]), -pose3D(order(3), [14 26]), [colorOption '-'], 'LineWidth',5);
    plot3(pose3D(order(1), [26 27]), pose3D(order(2), [26 27]), -pose3D(order(3), [26 27]), [colorOption '-'], 'LineWidth',5);
    plot3(pose3D(order(1), [27 28]), pose3D(order(2), [27 28]), -pose3D(order(3), [27 28]), [colorOption '-'], 'LineWidth',5);
    plot3(pose3D(order(1), [1 2]), pose3D(order(2), [1 2]), -pose3D(order(3), [1 2]), [colorOption '-'], 'LineWidth',5);
    plot3(pose3D(order(1), [2 3]), pose3D(order(2), [2 3]), -pose3D(order(3), [2 3]), [colorOption '-'], 'LineWidth',5);
    plot3(pose3D(order(1), [3 4]), pose3D(order(2), [3 4]), -pose3D(order(3), [3 4]), [colorOption '-'], 'LineWidth',5);
    plot3(pose3D(order(1), [1 7]), pose3D(order(2), [1 7]), -pose3D(order(3), [1 7]), [colorOption '-'], 'LineWidth',5);
    plot3(pose3D(order(1), [7 8]), pose3D(order(2), [7 8]), -pose3D(order(3), [7 8]), [colorOption '-'], 'LineWidth',5);
    plot3(pose3D(order(1), [8 9]), pose3D(order(2), [8 9]), -pose3D(order(3), [8 9]), [colorOption '-'], 'LineWidth',5);
    [ind, labels] = get_jointset('relevant');
    for i = 1:17
%         text(pose3D(order(1),ind(i)), pose3D(order(2),ind(i)), -pose3D(order(3), ind(i)), labels{i});
    end
    xlabel('X');
    ylabel('Y')
    zlabel('Z')
    grid on
    
    view(0,0);
    % plot3(pose3D(order(1), [10 9]), pose3D(order(2), [10 9]), pose3D(order(3), [10 9]), [colorOption '-']);
    % plot3(pose3D(order(1), [10 11]), pose3D(order(2), [10 11]), pose3D(order(3), [10 11]), [colorOption '-']);
%     axis equal;
else
    %  1.   Left ankle 1
    %  2.   Left knee 2
    %  3.   Left hip 3
    %  4.   Right ankle 4
    %  5.   Right knee 5
    %  6.   Right hip 6
    %  7.   Left wrist 7
    %  8.   Left elbow 8
    %  9.   Left shoulder 9
    %  10.  Right wrist 10
    %  11.  Rigth elbow 11
    %  12.  Right shoulder 12
    %  13.  Neck 13
    %  14.  Head top 14
    disp('here')
   labels = {'Lank', 'Lkne', 'Lhip', 'Rank', 'Rkne', 'Rhip', 'Lwri', 'Lelb', 'Lsho', 'Rwri', 'Relb', 'Rsho', 'neck', 'head', 'HipCenter'};
%    labels =  {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
%                          'Lsho', 'Lelb', 'Lwri', ...
%                          'Rhip', 'Rkne', 'Rank', ...
%                          'Lhip', 'Lkne', 'Lank', 'HipCenter'};
                     
%     labels = {'Right ankle', 'Right knee', 'Right hip', 'Left hip', 'Left knee', 'Left ankle', 'Right wrist', 'Right elbow', 'Right shoulder', 'Left shoulder', 'Left elbow', 'Left wrist', 'Neck', 'Head top', 'Hip Center'};
    connections = zeros(15, 15);
    pose3D(15, :) = (pose3D(3, :) + pose3D(6, :)) / 2;
     
    % foot right
    connections(4, 5) = 1;
    connections(5, 6) = 1;
    
    % foot left
    connections(1, 2) = 1;
    connections(2, 3) = 1;
    
    % arm right
    connections(10, 11) = 1;
    connections(11, 12) = 1;
    
    % arm left
    connections(7, 8) = 1;
    connections(8, 9) = 1;
    
    % head
    connections(13, 14) = 1;
    
    % spine
    connections(15, 13) = 1;
    
    % hips
    connections(15, 3) = 1;
    connections(15, 6) = 1;
    
    
    % shoulders
    connections(13, 12) = 1;
    connections(13, 9) = 1;
%     
%     % foot right
%     connections(1, 2) = 1;
%     connections(2, 3) = 1;
%     
%     % foot left
%     connections(4, 5) = 1;
%     connections(5, 6) = 1;
%     
%     % arm right
%     connections(7, 8) = 1;
%     connections(8, 9) = 1;
%     
%     % arm left
%     connections(10, 11) = 1;
%     connections(11, 12) = 1;
%     
%     % head
%     connections(13, 14) = 1;
%     
%     % spine
%     connections(15, 13) = 1;
%     
%     % hips
%     connections(15, 3) = 1;
%     connections(15, 4) = 1;
    
    for i = 1:15
        for j = 1:15
            if(connections(i,j))
                plot(pose3D([i j], 1), pose3D([i j], 2),  [colorOption '-'], 'LineWidth',5);
            end
        end
    end
    plot(pose3D(:, 1), pose3D(:, 2), 'bd', 'MarkerSize', 8,  'MarkerFaceColor',[1,1,0.5]);
    for i = 1:15
%         text(pose3D(i,1), pose3D(i,2), labels{i}, 'FontSize', 7, 'FontWeight', 'bold', 'Color', [1 1 0]);
    end
    % plot3(pose3D(order(1), [10 9]), pose3D(order(2), [10 9]), pose3D(order(3), [10 9]), [colorOption '-']);
    % plot3(pose3D(order(1), [10 11]), pose3D(order(2), [10 11]), pose3D(order(3), [10 11]), [colorOption '-']);
%     axis equal;
end
end