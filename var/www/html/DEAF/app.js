const audioContext = new AudioContext();
var analyser = audioContext.createAnalyser();
const canvas = document.getElementById('canvas');
const canvasCtx = canvas.getContext('2d');

const background = document.getElementById('background')


const loading = document.createElement('img')
loading.id = 'loading'
loading.src = 'brugreen.gif'


const audiodiv = document.querySelector('.audiodiv')
const slider = document.querySelector('.slider')
const slider2 = document.querySelector('.slider2')
const slider3 = document.querySelector('.slider3')
const slider5 = document.querySelector('.slider5')
const slider6 = document.querySelector('.slider6')


const primaryGainControl = audioContext.createGain()
primaryGainControl.gain.setValueAtTime(1, 0);
primaryGainControl.connect(audioContext.destination);

const primaryGainControl2 = audioContext.createGain()
primaryGainControl2.gain.setValueAtTime(0.1, 0);
primaryGainControl2.connect(audioContext.destination);


const sampleFilter = audioContext.createBiquadFilter();
sampleFilter.type =  "highpass"
sampleFilter.frequency.value = 0;
sampleFilter.connect(analyser)


const primaryGainControl3 = audioContext.createGain()
primaryGainControl3.gain.setValueAtTime(0.1, 0);
primaryGainControl3.connect(sampleFilter);

const trackbutton = document.createElement('button')
trackbutton.innerText = 'play'
trackbutton.id = 'track'
audiodiv.appendChild(trackbutton)

const pausebutton = document.createElement('button')
pausebutton.innerText = 'pause'
pausebutton.id = 'pause'
audiodiv.appendChild(pausebutton)

const settings = document.getElementById('settings')
const levels = document.querySelector('.hiddenlevels')
let clicked = false
settings.addEventListener('click', async () => {
  if(clicked == false){
  levels.style.visibility = 'visible'
  clicked = true
  }
  else{
    levels.style.visibility = 'hidden'
    clicked = false
  }



})



const samplebutton = document.createElement('button')
samplebutton.innerText = 'sample'
samplebutton.addEventListener('click', async () => {
    
const canvas = document.getElementById('canvas');
const canvasCtx = canvas.getContext('2d');
canvasCtx.fillStyle = 'green';
canvasCtx.fillRect(10, 10, 150, 100);




})

//audiodiv.appendChild(samplebutton)
const oscGain = audioContext.createGain()
    oscGain.gain.setValueAtTime(0.5, 0);
    oscGain.connect(audioContext.destination)




const toneButton = document.createElement('button')
toneButton.innerText = "tone"
toneButton.addEventListener("click", () => {

})
//audiodiv.appendChild(toneButton)
let botarr = []
let toparr = []
let pitchno = []
let pitchavg = 0

let samplefreq = 0
let largestfreq = 0

let notearray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]




analyser.fftSize = 2048*8
var bufferLength = analyser.frequencyBinCount;
var dataArray = new Uint8Array(bufferLength);

analyser.getByteTimeDomainData(dataArray)

var freqdomain = new Uint8Array(analyser.frequencyBinCount)

let c = 2

function setVocals(){
  c = 1
}

function setPiano(){
  c = 2
}

function setSong(){
  c = 3
}

function setCello(){
  c = 4
}

function setRock(){
  c = 5
}

function resetc(){
  c = 0
}




function draw() {

    analyser.getByteTimeDomainData(dataArray);

    for(var i = 0; i < bufferLength; i++) {

      //console.log(v * 200/2)

  }
};



trackbutton.addEventListener('click', async () => {

  trackbutton.style.backgroundColor = 'blue'
  setTimeout(() => {
    trackbutton.style.backgroundColor = 'white'
  }, 1000);
// initiation
pausebutton.addEventListener('click', async () => {

  


  sampleSource.stop()
  sampleSource2.stop()

  trackbutton.style.backgroundcolor = 'blue'
  setTimeout(() => {
    trackbutton.style.backgroundcolor = 'white'
  }, 1000);

})

background.appendChild(loading)




    let soundBuffer = ''

    if(c == 1){
      const response = await fetch(response)
      soundBuffer = await response.arrayBuffer()


      
    }
    if(c == 2){
      const response = await fetch('scale.wav')
      soundBuffer = await response.arrayBuffer()
      
    }
    if(c == 3){
      const response = await fetch('solopiano.wav')
      soundBuffer = await response.arrayBuffer()
    }
    if(c == 4){
      const response = await fetch('cello.wav')
      soundBuffer = await response.arrayBuffer()
    }
    if(c == 5){
      const response = await fetch('carryon.wav')
      soundBuffer = await response.arrayBuffer()
    }
    if(c == 0){
      soundBuffer = await document.getElementById('input').files[0].arrayBuffer()
      
    }
    const sampleBuffer = await audioContext.decodeAudioData(soundBuffer)

    const sampleSource = audioContext.createBufferSource()
    const sampleSource2 = audioContext.createBufferSource()

    sampleSource.buffer = sampleBuffer
    sampleSource2.buffer = sampleBuffer

    sampleSource2.connect(primaryGainControl2)
    sampleSource.connect(primaryGainControl3);
    primaryGainControl3.gain.setValueAtTime(slider2.value/10, 0)


    
    setTimeout(() => {
      sampleSource2.start()
    }, slider5.value*10); 
    //console.log(slider5.value*20) 
    background.removeChild(loading)
    sampleSource.start()


    const toneOscillator = audioContext.createOscillator();
  toneOscillator.frequency.setValueAtTime(0, 0)
  toneOscillator.type = "sine"
  toneOscillator.connect(oscGain);
  toneOscillator.frequency.setValueAtTime(120, 0)
  toneOscillator.start()
  toneOscillator.stop(audioContext.currentTime + 200)


const tune = document.querySelector('.slider4').value

const tunedial = document.querySelector('#tune')
const dbdial = document.querySelector('#db')

  //track code

  rpt()
  function rpt(){
    setTimeout(() => {


      

      const now = audioContext.currentTime;
        
      sampleFilter.frequency.value = (slider3.value*10)*3;

        let newvol = 0.01

        analyser.getByteTimeDomainData(dataArray);
        analyser.getByteFrequencyData(freqdomain)
        //console.log(analyser)


          let i;
          let max = dataArray[0];
          let frq = 0
          

          for (i = 1; i < dataArray.length; i++) {
              if (dataArray[i] > max){
                  max = dataArray[i];
                  frq = i
                  //console.log(freqdomain[i])

                 //frq = freqdomain[index]


              }

          }

        
        //console.log(frq)
        //console.log(bufferLength)

        newvol = (((max / 128.0)*50)-50)/10
        //console.log(newvol)
        if(newvol < 0 == true){
          newvol = newvol * -1
        }
        //THRESHHOLD
//console.log(newvol)
        if(newvol > 0.2){



          let tune = document.querySelector('.slider4').value/5
          for (i = 1; i < (freqdomain.length - 1024); i++) {

            if(botarr.length <= 200){
                botarr.push(freqdomain[i])
            }

            }

            if(botarr.length > 198){
              //
              botarr.forEach((item2, index2) => {

            for (var y = 0; y < botarr.length; y++) {
                if (largestfreq < botarr[y] ) {
                  if(botarr[y] > 40){
                    largestfreq = botarr[y];
                  }


                }
            }
            
            let prevfreq = 0
            if(botarr.indexOf(largestfreq) > 1){
              //if(samplefreq > samplefreq - 10 && samplefreq < samplefreq + 10)



             
              


            if(botarr.indexOf(largestfreq)*1.36263541667 != samplefreq){
            samplefreq = (botarr.indexOf(largestfreq)*1.36263541667)+13
            if(samplefreq < 80 ){
              samplefreq = samplefreq*2
            }
            if(samplefreq > 160 ){
              samplefreq = samplefreq/2
            }



              
              //PITCH THRSH
              //TOP
           
              if(samplefreq!=prevfreq){
              
              if(samplefreq/4 > 120 ){

                toneOscillator.frequency.setValueAtTime(((samplefreq/4)/2)+tune, now )
                prevfreq = samplefreq
                
                tunedial.innerHTML = `<p> ${samplefreq.toFixed(2)} HZ <p>`
              }
              else{

                if(samplefreq/4 < 80){

                  toneOscillator.frequency.setValueAtTime(((samplefreq/4)*2)+tune, now )
                  prevfreq = samplefreq
        
                  tunedial.innerHTML = `<p> ${samplefreq.toFixed(2)} HZ <p>`
                }
                else{
                 toneOscillator.frequency.setValueAtTime((samplefreq/4)+tune, now )
                 prevfreq = samplefreq
     
                 tunedial.innerHTML = `<p> ${samplefreq.toFixed(2)} HZ <p>`
                }
              }

            

            }
            }


            
          }
              })
              //console.log(samplefreq)
              


          }


        largestfreq = 0
        botarr = []
        samplefreq = 0

          newvol = newvol*(slider.value/100)
          primaryGainControl2.gain.setValueAtTime(slider2.value/150, now)
          
          //pitchno.push(frq)
          pitchno.push(freqdomain)
            if(newvol == 0 || newvol == 0.4609375){
              newvol = 0.01
              oscGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.15)
              dbdial.innerHTML = `<p> 0 <p>`
              //oscGain.gain.setValueAtTime(newvol, 0);
            }
            if(newvol < 0.9){

            oscGain.gain.exponentialRampToValueAtTime(newvol, now + 0.15)
            dbdial.innerHTML = `<p> ${(newvol*5).toFixed(2)} DB <p>`
            }
            //VOLUME MAXIMUM THRSH
            else{
              oscGain.gain.exponentialRampToValueAtTime(0.8, now + 0.15)
              dbdial.innerHTML = `<p> ${(newvol*5).toFixed(2)} DB <p>`
            }

        }
        else{
          pitchno = []
          oscGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.15)
          dbdial.innerHTML = `<p> ${(newvol*5).toFixed(2)} DB<p>`
        }

        if(freqdomain.length >= 100){
          pitchavg =  (pitchno.reduce((partialSum, a) => partialSum + a, 0))/pitchno.length
          //console.log(pitchavg)
        }


        rpt()
      }, 50);
  }

  function draw() {
    const drawVisual = requestAnimationFrame(draw);



    canvasCtx.fillStyle = '#000000';
canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);
canvasCtx.lineWidth = 2;
canvasCtx.strokeStyle = '#ffffff';
canvasCtx.beginPath();
const sliceWidth = WIDTH * 1.0 / bufferLength;
let x = 0;
let y = 0;

for (let i = 0; i < bufferLength; i++) {

  const v = dataArray[i] / 128.0;
  const y = v * HEIGHT/2;

  if (i === 0) {
    canvasCtx.moveTo(x, y);
  } else {
    canvasCtx.lineTo(x, y);
  }

  x += sliceWidth;
}

canvasCtx.lineTo(canvas.width, canvas.height/2);
canvasCtx.stroke();
};

draw();



})

const WIDTH = 300
const HEIGHT =150
canvasCtx.clearRect(0, 0, WIDTH, HEIGHT);






//Make the bass glide to every 5 or so numbers of final calculations

