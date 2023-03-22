var ensembleList = Array();
var boxplotList = Array();
var map1List = Array();
var map2List = Array();
var states = ['AZ', 'CO', 'GA', 'MI', 'MN', 'NC', 'OH', 'PA', 'TX', 'WI']
for (var i = 0; i < 10; i++) {
    ensembleList[i] = new Image(800, 600);
    ensembleList[i].src = "./output/" + states[i] + "2011/ensemble.png";
    boxplotList[i] = new Image(1400, 600);
    boxplotList[i].src = "./output/" + states[i] + "2011/boxplot.png";
    map1List[i] = new Image(1400, 700);
    map1List[i].src = "./output/" + states[i] + "2011/map1.png";
    map2List[i] = new Image(1400, 700);
    map2List[i].src = "./output/" + states[i] + "2011/map2.png";
}

function switchImage() {
    var selectedImage = document.myForm.switch.options[document.myForm.switch.selectedIndex].value;
    document.ensemble.src = ensembleList[selectedImage].src;
    document.boxplot.src = boxplotList[selectedImage].src;
    document.map1.src = map1List[selectedImage].src;
    document.map2.src = map2List[selectedImage].src;
}