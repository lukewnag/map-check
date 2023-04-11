var ensembleList = Array();
var boxplotList = Array();
var map1List = Array();
var map2List = Array();
var states = ['AZ', 'CO', 'GA', 'IL', 'MI', 'MN', 'NC', 'OH', 'PA', 'SC', 'TX', 'WI']
for (var i = 0; i < 12; i++) {
    ensembleList[i] = new Image(800, 600);
    ensembleList[i].src = "./output/" + states[i] + "/ensemble.png";
    boxplotList[i] = new Image(1400, 600);
    boxplotList[i].src = "./output/" + states[i] + "/boxplot.png";
    map1List[i] = new Image(1400, 700);
    map1List[i].src = "./output/" + states[i] + "/map1.png";
    map2List[i] = new Image(1400, 700);
    map2List[i].src = "./output/" + states[i] + "/map2.png";
}

var stateFullNames = ['Arizona', 'Colorado', 'Georgia', 'Illinois', 'Michigan', 'Minnesota', 'North Carolina',
    'Ohio', 'Pennsylvania', 'South Carolina', 'Texas', 'Wisconsin']
function switchImage() {
    var selectedImage = document.stateSelector.switch.options[document.stateSelector.switch.selectedIndex].value;
    document.ensemble.src = ensembleList[selectedImage].src;
    document.boxplot.src = boxplotList[selectedImage].src;
    document.map1.src = map1List[selectedImage].src;
    document.map2.src = map2List[selectedImage].src;
    document.getElementById('selection').innerHTML = stateFullNames[selectedImage];
}

var coll = document.getElementsByClassName("collapsible");
for (var i = 0; i < coll.length; i++) {
    coll[i].addEventListener("click", function() {
        this.classList.toggle("active");
        var content = this.nextElementSibling;
        if (content.style.maxHeight){
            content.style.maxHeight = null;
        } else {
            content.style.maxHeight = content.scrollHeight + "px";
        }
    });
}

const mobileBtn = document.getElementById('mobile-cta')
    nav = document.querySelector('nav')
    mobileBtnExit = document.getElementById('mobile-exit')

    mobileBtn.addEventListener('click', () => {
        nav.classList.add('menu-btn')
    })
    mobileBtnExit.addEventListener('click', () => {
        nav.classList.remove('menu-btn')
    })