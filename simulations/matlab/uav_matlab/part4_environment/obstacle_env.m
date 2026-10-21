classdef obstacle_env
% OBSTACLE_ENV  3D obstacle environment matching Paper 1's setup
% Space: 2000 x 2000 x 500 m, cylinders (towers) + cuboids (buildings)

    properties
        obstacles   % struct array: type, center [x y z_base], dims, buffer
        space_x = 2000
        space_y = 2000
        space_z = 500
        start_pt
        goal_pt
    end

    methods
        function obj = obstacle_env(start_pt, goal_pt)
            obj.start_pt = start_pt;
            obj.goal_pt  = goal_pt;
            obj.obstacles = struct('type',{},'center',{},'dims',{},'buffer',{});
        end

        function obj = add_cylinder(obj, cx, cy, z_base, radius, height, buffer)
            if nargin < 7, buffer = 20; end
            n = numel(obj.obstacles) + 1;
            obj.obstacles(n).type   = 'cylinder';
            obj.obstacles(n).center = [cx, cy, z_base];
            obj.obstacles(n).dims   = [radius, height];
            obj.obstacles(n).buffer = buffer;
        end

        function obj = add_cuboid(obj, cx, cy, z_base, lx, ly, height, buffer)
            if nargin < 8, buffer = 20; end
            n = numel(obj.obstacles) + 1;
            obj.obstacles(n).type   = 'cuboid';
            obj.obstacles(n).center = [cx, cy, z_base];
            obj.obstacles(n).dims   = [lx, ly, height];
            obj.obstacles(n).buffer = buffer;
        end

        function tf = is_collision(obj, p)
        % p = [x y z], returns true if inside any obstacle (+ buffer)
            tf = false;
            for i = 1:numel(obj.obstacles)
                o = obj.obstacles(i);
                b = o.buffer;
                switch o.type
                    case 'cylinder'
                        r = o.dims(1) + b;
                        h = o.dims(2) + b;
                        horiz_d = norm(p(1:2) - o.center(1:2));
                        if horiz_d <= r && p(3) >= o.center(3)-b && p(3) <= o.center(3)+h
                            tf = true; return;
                        end
                    case 'cuboid'
                        lx = o.dims(1)/2 + b;
                        ly = o.dims(2)/2 + b;
                        hz = o.dims(3) + b;
                        if abs(p(1)-o.center(1)) <= lx && ...
                           abs(p(2)-o.center(2)) <= ly && ...
                           p(3) >= o.center(3)-b && p(3) <= o.center(3)+hz
                            tf = true; return;
                        end
                end
            end
        end

        function plot_obstacles(obj, ax)
            if nargin < 2, ax = gca; end
            hold(ax, 'on');
            for i = 1:numel(obj.obstacles)
                o = obj.obstacles(i);
                switch o.type
                    case 'cylinder'
                        obj.draw_cylinder(ax, o);
                    case 'cuboid'
                        obj.draw_cuboid(ax, o);
                end
            end
        end
    end

    methods (Access = private)
        function draw_cylinder(~, ax, o)
            r = o.dims(1); h = o.dims(2);
            [X,Y,Z] = cylinder(r, 20);
            Z = Z*h + o.center(3);
            X = X + o.center(1);
            Y = Y + o.center(2);
            surf(ax, X, Y, Z, 'FaceColor',[0.8 0.6 0.4], ...
                'FaceAlpha',0.6, 'EdgeColor','none');
        end

        function draw_cuboid(~, ax, o)
            lx = o.dims(1)/2; ly = o.dims(2)/2; hz = o.dims(3);
            cx = o.center(1); cy = o.center(2); z0 = o.center(3);
            v = [cx-lx cy-ly z0;    cx+lx cy-ly z0;
                 cx+lx cy+ly z0;    cx-lx cy+ly z0;
                 cx-lx cy-ly z0+hz; cx+lx cy-ly z0+hz;
                 cx+lx cy+ly z0+hz; cx-lx cy+ly z0+hz];
            faces = [1 2 3 4; 5 6 7 8; 1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8];
            patch(ax, 'Vertices',v, 'Faces',faces, ...
                'FaceColor',[0.6 0.6 0.8], 'FaceAlpha',0.6, 'EdgeColor','k');
        end
    end
end