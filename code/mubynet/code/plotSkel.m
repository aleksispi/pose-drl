function plotSkel( pose3D, colorOption, invPoints)

if(nargin < 3)
    invPoints = [];
end


if (size(pose3D, 1) ~= 32 && size(pose3D, 2) == 3)
    buff_large = zeros(32, 3);
    [ind, ~] = get_jointset('relevant');
    buff_large(ind, :) = pose3D;
    pose3D = buff_large';
end;

hold on;

order = [1 3 2];
for i = 1:size(invPoints, 2)
    plot3(invPoints(order(1), i), invPoints(order(2), i), -invPoints(order(3), i), 'w*');
end
if(size(pose3D, 1) == 3)
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

    xlabel('X');
    ylabel('Y')
    zlabel('Z')
    grid on
    
    view(4,2);
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

end
end