function rgbs=hsvf(h,s,v) 

    if nargin == 1
        s = h(:,2);
        v = h(:,3);
        h = h(:,1);
    end
    
    if all(rem(h,1) == 0)
        h = h/360;
    end

    rgbs = hsv2rgb([h,s,v]);
    
end