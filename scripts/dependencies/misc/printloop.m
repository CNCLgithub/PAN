function printloop(index, limit, ntabs)

    if ~exist("ntabs","var") || (ntabs == 0),tabs = '';
    else, tabs = repmat('\t',1,ntabs); end
           
    str = strcat(tabs, "%d/%d\n");   
    fprintf(str,index,limit)


end