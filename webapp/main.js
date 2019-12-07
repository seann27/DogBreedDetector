// test that file is loaded properly
$(document).ready(function() {
    // document.getElementById("result").textContent="ello govna!";
    $.getJSON('default.json',function(){
        $.each(data, function(i,img){
            $('ul#images').append('<li>'+img.imageName+'</li>');
        });
    });
});
