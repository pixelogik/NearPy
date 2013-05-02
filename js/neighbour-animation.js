var myShapeA = new Circle(0, 0, 50).addTo(stage).attr('fillColor', 'red');
var myShapeB = new Circle(0, 50, 50).addTo(stage).attr('fillColor', 'blue');
var myAnimation = new Animation('1s', {
  x: 100,
  fillColor: 'green'
});
myAnimation.addSubjects([myShapeA, myShapeB]);
myAnimation.play();