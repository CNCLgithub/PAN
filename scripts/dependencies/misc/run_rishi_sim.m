function [pass,coors] = run_rishi_sim(x0,y0, h0, sp0 , bounce_lims)

    nstates = numel(x0);
    coors.x = cell(nstates,1);
    coors.y = coors.x;

    xrange = [-10 10];
    yrange = [-10 10];

    % Min/Max for Rishi's perameters in ms
    mn = 624.9;
    mx = 1874.7;

    % occluder position for Rishi's trials
    occluder_width = 0.25;

    mxtime = ceil(mx*2); % both conditions vis/occ

    width = diff(xrange);
    
    occ_x = ((1 - occluder_width)*width) + xrange(1);

    deg_per_ms = sp0*(1/1000);

    h0   = deg2rad(h0);
    dx0s = cos(h0).*deg_per_ms;
    dy0s = sin(h0).*deg_per_ms;
        
    game_end = xrange(2);

    
    pass  = false(nstates,1);

    for i = 1:nstates

        [x,y] = deal(nan(mxtime,1));

        x(1) = x0(i);
        y(1) = y0(i);
        dx0  = dx0s(i);
        dy0  = dy0s(i);
    
        count = 2; bounce = 0;
    
        while x(count-1) <= game_end 
                
                x(count) = x(count-1) + dx0; 
                y(count) = y(count-1) + dy0;
        
                if y(count) <= yrange(1) || y(count) >= yrange(2)
                    dy0 = -dy0;
                    bounce = bounce+1;
                end
            
                if bounce == bounce_lims(2)+1,return,end
    
                if x(count) >= xrange(2), break, end
    
                count = count+1;
        end
    
        if bounce == bounce_lims(1)-1,return,end
    
        x(count+1:end) = [];
        y(count+1:end) = [];
    
        vis  = x < occ_x;
        nvis = sum(vis);
        nocc = sum(~vis);
        
        coors.x{i} = x;
        coors.y{i} = y;

        if prod([nvis nocc] >= mn) && prod([nvis nocc] <=mx), pass(i) = true; end
    end
end