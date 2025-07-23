function c = pong_colors

    f = @(h,s,v) hsv2rgb([h,s,v]);


    c.brain.color     = f(000/360,  1 , 0.7  );         
    c.prnn.color      = f(180/360, .6 , 0.8  );        
    c.random.color    = f(230/360, .1 , 0.4  );          
    c.nextTP.color    = f(120/360, .4 , 0.4  );          
    c.paddle.color    = f(045/360, .5 , 0.6  );   
    c.allsim2.color   = f(120/360, .4 , 0.4  );          
    c.allsim.color    = f(090/360, .5 , 0.7  );
    c.vissim.color    = f(060/360, .5 , 0.8  );  
    c.nosim.color     = f(045/360, .5 , 0.6  ); 

    c.brain.hue     = 000/360;     
    c.prnn.hue      = 180/360;   
    c.random.hue    = 230/360;      
    c.nextTP.hue    = 120/360;      
    c.paddle.hue    = 045/360;      
    c.allsim2.hue   = 120/360;      
    c.allsim.hue    = 090/360;      
    c.vissim.hue    = 060/360;      
    c.nosim.hue     = 045/360;      
end



function sandbox
%%
figure
cimage([
f(000/360,  1 , 0.7  ) % brain              
f(180/360, .6 , 0.8  ) % prnn              
f(230/360, .1 , 0.4  ) % random                
f(120/360, .4 , 0.4  ) % nextTP                
f(045/360, .5 , 0.6  ) % paddle         
f(120/360, .4 , 0.4  ) % allsim2                
f(090/360, .5 , 0.7  ) % allsim         
f(060/360, .5 , 0.8  ) % vissim         
f(045/360, .5 , 0.6  ) % nosim         
],'axis',gca)


end