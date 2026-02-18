// JavaScript functions for showing/hiding forms and handling login


if (window.location.pathname === '/') {
    window.location.href = '/jav';
}


let isAuthenticated = false
let using = ''
let isPro = false
let mus = false
let currentVideoId = '' // Global variable to store the video ID
let videolabels = ''
let originalLengthInSeconds = 10
let currentZoomLevel = 1;


let splitScenes = false;
let useManualDuration   = false;

let globalSceneChanges = [];
let syncToSceneChanges = true;

let showMillisecondTicks = false;




const loading = document.getElementById('loading')





// First, get a reference to the text input element
let currentSlide = 0;








// Listen for scroll events



window.onload = function() {
    const isAuth = localStorage.getItem('isAuth') === 'true';
    const username = localStorage.getItem('username');
    const substatus = localStorage.getItem('ispro');
    const paydiv = document.getElementsByClassName('paycontainer')
    using =  localStorage.getItem('username');


    if(isAuth === true){
        document.querySelector('.guest').style.display = 'none';


    }
    else{
        //  window.location.href = "login.html"
    }



    const texturedText = document.querySelector('.textured-text');
  let animationEnabled = false;
  let animationTimer;

  // Function to enable animation when scrolling


    



    


    if (window.location.pathname == '/plans.html') {
        // Execute specific code for home.html
        if(isAuth === false){
            document.querySelector('.notsubbed').style.display ='none'
            document.querySelector('.notsignedin').style.display ='block'
            
        }
    }


    if (isAuth && username) {


        isAuthenticated = true

        toggleSignOutButton();
        document.getElementById('loginbutton').style.display = 'none';

        if(substatus == 'Pro+'){
            isPro = true




            
        }


        if (window.location.pathname == '/plans.html') {
            // Execute specific code for home.html

        if(substatus == 'Pro+'){
            isPro = true

   
            document.querySelector('.notsubbed').style.display ='none'
            document.querySelector('.subbed').style.display ='block'
            document.querySelector('.notsignedin').style.display ='none'

            if (window.location.pathname == '/plans.html') {
                //paydiv.style.visibility = 'hidden'
                }
            

            
        }
    }
  
    



        try {
            

            

        // User is authenticated, update the UI accordingly
     
        document.getElementById('user-username').textContent = username;
        document.getElementById('user-subscription-status').textContent = substatus;

        document.getElementById('user-info').style.display = 'block';
        document.getElementById('register-form').style.display = 'none';
        document.getElementById('login-form').style.display = 'none';
        document.getElementById('signupbutton').style.display = 'none';
        document.getElementById('loginbutton').style.display = 'none';


        toggleSignOutButton();

    } catch (error) {

    }

    

        


        // Add any other UI updates needed for an authenticated user
    }else{

        if(window.location.pathname != '/home.html'){
        //document.getElementById('user-info').style.display = 'none';
        }
    }







};

if (isPro === true) {

    // Select the container
    const container = document.querySelector('.paycontainer');

    // Select and remove the PayPal sign-up div (assuming it has an ID 'paypal-signup')
    const paypalDiv = container.querySelector('.paypalbuttons');
    if (paypalDiv) {
        container.removeChild(paypalDiv);
    }

    // Create a new div to show subscribed status
    const subscribedDiv = document.createElement('div');
    subscribedDiv.textContent = 'You are subscribed. Thank you!';
    
    // Append the new div to the container
    container.appendChild(subscribedDiv);
    document.getElementsByClassName('notsubbed').style.display = 'none'
}
else{ 


}
if (window.location.pathname == '/plans.html') {
document.body.style.marginLeft = '50px';
document.querySelector('#background').style.position = 'relative';
document.querySelector('#background').style.right = '70px';




}



let key = false
if (window.location.pathname == '/doseedo1.html' ) {

    // Example usage
//createDownloadLinks(['ComoSomos.wav', '/download/c1d2f10b-a275-4145-b68e-b20f14d32149/1.wav', '/download/c1d2f10b-a275-4145-b68e-b20f14d32149/2.wav', '/download/c1d2f10b-a275-4145-b68e-b20f14d32149/3.wav', '/download/c1d2f10b-a275-4145-b68e-b20f14d32149/4.wav', '/download/c1d2f10b-a275-4145-b68e-b20f14d32149/5.wav', '/download/c1d2f10b-a275-4145-b68e-b20f14d32149/6.wav', '/download/c1d2f10b-a275-4145-b68e-b20f14d32149/7.wav', '/download/c1d2f10b-a275-4145-b68e-b20f14d32149/8.wav', '/download/c1d2f10b-a275-4145-b68e-b20f14d32149/9.wav', '/download/c1d2f10b-a275-4145-b68e-b20f14d32149/10.wav', '/download/c1d2f10b-a275-4145-b68e-b20f14d32149/11.wav', '/download/c1d2f10b-a275-4145-b68e-b20f14d32149/12.wav']); // Add your file paths here

const sceneChanges = [0, 1, 4]



//createDownloadLinks(['/download/44dc5e49-2c2c-4bd9-b2ce-eaf80230e9cd/1.wav', '/download/44dc5e49-2c2c-4bd9-b2ce-eaf80230e9cd/2.wav', '/download/44dc5e49-2c2c-4bd9-b2ce-eaf80230e9cd/3.wav'])



document.addEventListener('DOMContentLoaded', function() {



    var textInput = document.getElementById('description-input');

// Define a function that will be called whenever the input event occurs
function handleInput(event) {
    console.log('Input event occurred! Value is:', event.target.value);
    document.querySelector('#gensfx').style.opacity = '100%'
    document.querySelector('#gensfx').style.pointerEvents = 'auto'
    document.querySelector('#genmus').style.opacity = '100%'
    document.querySelector('#genmus').style.pointerEvents = 'auto'
}



// Add the event listener to the text input element
textInput.addEventListener('input', handleInput);{


}


if (screen.width <= 700) {



    document.body.style.transform = 'scale(1.8)'
    document.body.style.marginTop = '200%'
    document.body.style.marginLeft = '200px'

  

    }

    document.body.style.marginLeft = '50px';
    document.querySelector('#background').style.position = 'relative';
    // document.querySelector('#background').style.right = '10%';
    //document.querySelector('.cues').style.display = 'none';
    document.querySelector('.buttons').style.display = 'block';
    // document.querySelector('#background').style.width = '120%';
    var div = document.querySelector(".input-group1")
    var div1 = document.querySelector(".input-group2")
    var div2 = document.querySelector(".input-group3")


    // div.style.display = 'none';
    // div1.style.display = 'none';
    // div2.style.display = 'none';


  });
}

function showSignUp() {
    document.getElementById('register-form').style.display = 'block';
    document.querySelector('.formdiv').style.display = 'block';
    //document.querySelector('.formdiv').style.display = 'block';
    document.getElementById('login-form').style.display = 'none';
    document.getElementById('user-info').style.display = 'none';
    document.body.style.bottom = '100px'
    document.getElementById('background').style.opacity = '0.1'
}



function showSignIn() {
    document.getElementById('register-form').style.display = 'none';
    document.querySelector('.formdiv').style.display = 'block';
    //document.querySelector('.formdiv').style.display = 'block';
    document.getElementById('login-form').style.display = 'block';
    document.getElementById('user-info').style.display = 'none';

    document.getElementById('background').style.opacity = '0.1'


}
function exitregistration(){
    document.getElementById('register-form').style.display = 'none';
    document.getElementById('login-form').style.display = 'none';
    document.getElementById('background').style.opacity = '1'
    document.querySelector('.formdiv').style.display = 'none';


}

function loginUser() {
    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;
    const user = { username: username, tokens: 10 };
    document.getElementById('user-username').textContent = user.username;
    document.getElementById('user-info').style.display = 'block';
    document.getElementById('register-form').style.display = 'none';
    document.getElementById('login-form').style.display = 'none';
    document.querySelector('.guest').style.display = 'none';

}

function registerUser() {
    const username = document.getElementById('register-username').value;
    const email = document.getElementById('register-email')
    const password = document.getElementById('register-password').value;
    alert(`User registered successfully!\nUsername: ${username}\nEmail: ${email}`);
}
function nextstep() {
    const username = document.getElementById('register-username').style.display = 'inline';
    const password = document.getElementById('register-password').style.display = 'inline'
    const email = document.getElementById('register-email')
    email.style.display = 'none'
        const nextbutton = document.getElementById('nextbutton').style.display = 'none'
        const registerbutton = document.getElementById('registerbutton').style.display = 'inline'
        const backbutton = document.querySelector("#backbutton").style.display = 'inline'
        const emaildisplay = document.querySelector("#emaildisplay")
        emaildisplay.innerHTML = `<p>${email.value}</p>`
}
function backtoemail() {
    const username = document.getElementById('register-username').style.display = 'none';

    const password = document.getElementById('register-password').style.display = 'none'
    const email = document.getElementById('register-email').style.display = 'inline'
        const nextbutton = document.getElementById('nextbutton').style.display = 'inline'
        const registerbutton = document.getElementById('registerbutton').style.display = 'none'
        const emaildisplay = document.querySelector("#emaildisplay").innerHTML = ''
             const backbutton = document.querySelector("#backbutton").style.display = 'none'
}


if (window.location.pathname == '/doseedo1.html') {


    const midiscales = {
        "prompt1": "11 Cmaj9.mid",
        "prompt2": "10 Cm9.mid",

    };


    const melodySelect = document.getElementById('prompt-select');
    const keySelect = document.getElementById('prompt-select2');
    let selectedMelody = melodySelect.value;
    let selectedKey = keySelect.value;
    
// Get the selected melody and key

    function handleSelectionChange() {
        selectedMelody = melodySelect.value;
        selectedKey = keySelect.value;
        


        key = true
        console.log(selectedKey)
        console.log(selectedMelody)
        console.log(key)
    }



    function generateAudio() {

    //updateStatus('Generating audio...');
    const descriptionInput = document.getElementById('description-input');
    const durationInput = document.getElementById('duration-input');
    const labels = descriptionInput.value.split(', ');
    const promptSelect = document.getElementById('prompt-select');
    const prompt = promptSelect.value;
    const duration = durationInput.value;
    const formData = new FormData();
    console.log(descriptionInput.value);
    formData.append('labels', JSON.stringify(videolabels));
    formData.append('prompt', prompt);
    formData.append('duration', duration);
    formData.append('video_output', descriptionInput.value);
    formData.append('request_type', 'normal');
    loading.style.visibility = 'visible'



    document.querySelector('#genmus').style.pointerEvents = 'none'
    document.querySelector('#genmus').style.opacity = '20%'

    document.querySelector('#gensfx').style.opacity = '20%'
    document.querySelector('#gensfx').style.pointerEvents = 'none'

    fetch('https://doseedo.com/generate', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data && data.task_id) {
            pollTaskStatus(data.task_id, createDownloadLinks);
        } else {
            console.log('No task ID received');
            updateStatus('Failed to start audio generation task.');
            loading.style.visibility = 'hidden'
            document.querySelector('#genmus').style.pointerEvents = 'auto'
            document.querySelector('#genmus').style.opacity = '100%'
        
            document.querySelector('#gensfx').style.opacity = '100%'
            document.querySelector('#gensfx').style.pointerEvents = 'auto'
        }
    })
    .catch(error => {
        console.error('There was a problem with the fetch operation:', error);
        updateStatus('Error during audio generation.');
        loading.style.visibility = 'hidden'
        document.querySelector('#genmus').style.pointerEvents = 'auto'
        document.querySelector('#genmus').style.opacity = '100%'
    
        document.querySelector('#gensfx').style.opacity = '100%'
        document.querySelector('#gensfx').style.pointerEvents = 'auto'
    });
}


// let splitScenes = /* document.getElementById('splitScenesCheckbox').checked*/ 

    
function generateMusic() {
    updateStatus('Generating music…');
  
    // --- Gather inputs & build query params ---
    const descIn = document.getElementById('description-input').value;
    const duration = parseFloat(document.getElementById('duration-input').value);
    const labels = descIn.split(', ');
    const params = new URLSearchParams();
  
    params.set('labels', JSON.stringify(labels));
    params.set('split_scenes', splitScenes);
    params.set('video_output', descIn);
    params.set('request_type', 'melody');
  
    if (
      !useManualDuration &&
      Array.isArray(globalSceneChanges) &&
      globalSceneChanges.length > 1
    ) {
      const segDurs = [];
      for (let i = 0; i < globalSceneChanges.length - 1; i++) {
        segDurs.push(globalSceneChanges[i+1] - globalSceneChanges[i]);
      }
      segDurs.push(originalLengthInSeconds - globalSceneChanges.at(-1));
      params.set('scene_durations', JSON.stringify(segDurs));
    } else {
      params.set('duration', duration);
    }
  
    // --- UI loading state ---
    loading.style.visibility = 'visible';
    ['#genmus','#gensfx'].forEach(sel => {
      const btn = document.querySelector(sel);
      btn.style.pointerEvents = 'none';
      btn.style.opacity       = '20%';
    });
    const hideLoading = () => {
      loading.style.visibility = 'hidden';
      ['#genmus','#gensfx'].forEach(sel => {
        const btn = document.querySelector(sel);
        btn.style.pointerEvents = 'auto';
        btn.style.opacity       = '100%';
      });
    };
  
    // --- Fallback POST path ---
    function oneShotSend(melodyFile) {
      const fd = new FormData();
      for (const [k,v] of params) fd.append(k, v);
      if (melodyFile) fd.append('melody', melodyFile);
  
      fetch('https://doseedo.com/generate', {
        method: 'POST',
        body: fd
      })
      .then(r => r.json())
      .then(data => {
        if (!data.task_id) throw new Error('No task ID received');
        pollTaskStatus(data.task_id, createDownloadLinks);
        updateStatus('Music Successfully Generated');
      })
      .catch(err => {
        console.error(err);
        updateStatus('Error during music generation.');
      })
      .finally(hideLoading);
    }
  
    // --- SSE streaming branch ---
    if (splitScenes) {
      const url = new URL('https://doseedo.com/generate-stream');
      url.search = params.toString();
  
      const evtSrc = new EventSource(url.toString());
  
      evtSrc.addEventListener('segment', e => {
        const { index, url: fileUrl } = JSON.parse(e.data);
        // render that one segment immediately
        createDownloadLinks([fileUrl], [ globalSceneChanges[index] ]);
      });
  
      evtSrc.addEventListener('complete', () => {
        updateStatus('All segments generated');
        hideLoading();
        evtSrc.close();
      });
  
      evtSrc.onerror = err => {
        console.error('Streaming error', err);
        updateStatus('Error during streaming');
        hideLoading();
        evtSrc.close();
      };
  
    // --- Else: regular POST path ---
    } else {
      if (!key) {
        oneShotSend(null);
      } else {
        params.set('prompt', keySelect.value);
        fetch(midiscales[selectedMelody])
          .then(r => r.blob())
          .then(blob => {
            const melodyFile = new File([blob], 'melody.mid', { type: 'audio/midi' });
            oneShotSend(melodyFile);
          })
          .catch(err => {
            console.error('Failed to fetch melody:', err);
            oneShotSend(null);
          });
      }
    }
  }
  
  // You can leave your existing pollTaskStatus() in place for the one-shot path (no changes needed).
  
  
  






// console.log('settings clicked')

setTimeout(() => {
    // document.getElementById('settings').click()
}, 800);


function debounce(func, wait) {
    var timeout;

    return function() {
        var context = this, args = arguments;

        var later = function() {
            timeout = null;
            func.apply(context, args);
        };

        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}




}









function music(){
    console.log(mus);
    mus = true;
    console.log(mus);
}

function updateStatus(message) {
    document.getElementById('status').textContent = message;
}


async function uploadVideo() {
    updateStatus('Uploading Video');
    loading.style.visibility = 'visible';

    const videoInput = document.getElementById('videoFile');
    if (videoInput.files.length === 0) {
        alert('Please select a video file.');
        return;
    }

    // Disable buttons
    document.querySelector('#genmus').style.pointerEvents = 'none';
    document.querySelector('#genmus').style.opacity = '20%';
    document.querySelector('#gensfx').style.opacity = '20%';
    document.querySelector('#gensfx').style.pointerEvents = 'none';

    const videoFile = videoInput.files[0];
    const formData = new FormData();
    formData.append('file', videoFile);

    try {
        const response = await fetch('https://doseedo.com/uploadvideo/', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        updateStatus('Uploaded Video');

        if (!data.video_id || !data.task_id) throw new Error('Missing video_id or task_id');

        currentVideoId = data.video_id;
        console.log('Video uploaded:', currentVideoId);

        // 👇 Poll for processing status
        pollTaskStatus2(data.task_id, data.audio_url);

    } catch (error) {
        console.error('Error during video upload:', error);
        updateStatus('Error Uploading');
        loading.style.visibility = 'hidden';

        document.querySelector('#genmus').style.pointerEvents = 'auto';
        document.querySelector('#genmus').style.opacity = '100%';
        document.querySelector('#gensfx').style.opacity = '100%';
        document.querySelector('#gensfx').style.pointerEvents = 'auto';
    }
}



async function extractAudioFromVideo(videoFile) {
    return new Promise((resolve, reject) => {
        const video = document.createElement('video');
        video.src = URL.createObjectURL(videoFile);
        video.crossOrigin = 'anonymous';
        video.preload = 'auto';
        video.muted = true;

        const audioChunks = [];

        video.addEventListener('canplaythrough', () => {
            const stream = video.captureStream();
            const audioStream = new MediaStream(
                stream.getAudioTracks()
            );

            if (audioStream.getAudioTracks().length === 0) {
                reject(new Error("No audio tracks found in video."));
                return;
            }

            const mediaRecorder = new MediaRecorder(audioStream, {
                mimeType: 'audio/webm' // most compatible format
            });

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                resolve(audioBlob);
            };

            mediaRecorder.onerror = (e) => {
                reject(new Error("MediaRecorder failed: " + e.error?.name || e.message));
            };

            mediaRecorder.start();

            video.play();

            video.onended = () => {
                mediaRecorder.stop();
            };
        });

        video.onerror = () => reject(new Error("Failed to load video"));
    });
}



function bufferToWavBlob(buffer) {
    const numOfChan = buffer.numberOfChannels;
    const length = buffer.length * numOfChan * 2 + 44;
    const bufferArray = new ArrayBuffer(length);
    const view = new DataView(bufferArray);
    const channels = [];
    let offset = 0;
    let pos = 0;

    function setUint16(data) {
        view.setUint16(pos, data, true);
        pos += 2;
    }

    function setUint32(data) {
        view.setUint32(pos, data, true);
        pos += 4;
    }

    // Write WAVE header
    setUint32(0x46464952); // "RIFF"
    setUint32(length - 8);
    setUint32(0x45564157); // "WAVE"

    setUint32(0x20746d66); // "fmt " chunk
    setUint32(16); // length = 16
    setUint16(1); // PCM (uncompressed)
    setUint16(numOfChan);
    setUint32(buffer.sampleRate);
    setUint32(buffer.sampleRate * 2 * numOfChan); // avg. bytes/sec
    setUint16(numOfChan * 2); // block-align
    setUint16(16); // 16-bit (hardcoded)

    setUint32(0x61746164); // "data" - chunk
    setUint32(length - pos - 4);

    // Write interleaved PCM data
    for (let i = 0; i < numOfChan; i++) {
        channels.push(buffer.getChannelData(i));
    }

    while (pos < length) {
        for (let i = 0; i < numOfChan; i++) {
            let sample = Math.max(-1, Math.min(1, channels[i][offset])); // clamp
            view.setInt16(pos, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            pos += 2;
        }
        offset++;
    }

    return new Blob([view], { type: 'audio/wav' });
}




function handleVideoUploadSuccess(videoId) {
    // Save the video ID for later use
    currentVideoId = videoId;
    // Additional code for what to do after successful upload
}

function handleVideoUploadSuccess(videoId) {
    // Save the video ID for later use
    currentVideoId = videoId;
    // Additional code for what to do after successful upload
}

let sceneChanges = []

function pollTaskStatus2(taskId, audioUrl) {
    const statusInterval = setInterval(() => {
      fetch(`https://doseedo.com/task-status/${taskId}`)
        .then(response => response.json())
        .then(data => {
          updateStatus(data.status);
  
          if (data.status === 'SUCCESS') {
            clearInterval(statusInterval);
            console.log('Video analysis results:', data.result);
  
            const labels = data.result.labels || [];
            let sceneChanges = data.result.scene_changes || [];
            const objectTracking = data.result.object_tracking || {};
            const impacts = data.result.impacts || [];
  
            // collapse any rapid‐cut clusters into start/end only
            const collapseSceneChanges = (changes, threshold) => {
              if (changes.length === 0) return [];
              const out = [];
              let start = changes[0], end = start;
              for (let i = 1; i < changes.length; i++) {
                const t = changes[i];
                if (t - end < threshold) {
                  // still in the same short cluster
                  end = t;
                } else {
                  // cluster ended — push start/end
                  if (start === end) out.push(start);
                  else { out.push(start); out.push(end); }
                  // reset
                  start = end = t;
                }
              }
              // final cluster
              if (start === end) out.push(start);
              else { out.push(start); out.push(end); }
              return out;
            };
  
            // apply a 3‑second threshold
            sceneChanges = collapseSceneChanges(sceneChanges, 3);

            document.getElementById('muss').classList.add('selected');
  
            // Display labels
            const descriptionInput = document.getElementById('description-input');
            descriptionInput.value = labels.join(', ');
  
            globalSceneChanges = sceneChanges;
  
            console.log("Labels:", labels);
            console.log("Filtered Scene Change Timestamps:", sceneChanges);
            console.log("Object Tracking Data:", objectTracking);
            console.log("Detected Impacts:", impacts);
  
            // Enable UI
            loading.style.visibility = 'hidden';
            document.querySelector('#genmus').style.pointerEvents = 'auto';
            document.querySelector('#genmus').style.opacity = '100%';
            document.querySelector('#gensfx').style.pointerEvents = 'auto';
            document.querySelector('#gensfx').style.opacity = '100%';
  
            const serverAudioUrl = data.result.audio_url;
            if (serverAudioUrl) {
              console.log("Loading waveform from audio:", serverAudioUrl);
              createDownloadLinks([`https://doseedo.com${serverAudioUrl}`], sceneChanges, 'download-links3');
            } else {
              console.warn("No audio_url returned");
            }
  
            // Optional cue triggers
            if (sceneChanges.length > 0) {
              triggerCueingSystem(sceneChanges, impacts);
            }
  
          } else if (data.status === 'FAILURE') {
            clearInterval(statusInterval);
            throw new Error('Error in video processing');
          } else {
            console.log('Processing... Status:', data.status);
          }
        })
        .catch(error => {
          clearInterval(statusInterval);
          console.error('Error:', error);
          updateStatus('Error during task status check.');
          loading.style.visibility = 'hidden';
          document.querySelector('#genmus').style.pointerEvents = 'auto';
          document.querySelector('#genmus').style.opacity = '100%';
          document.querySelector('#gensfx').style.pointerEvents = 'auto';
          document.querySelector('#gensfx').style.opacity = '100%';
        });
    }, 3000);
  }
  


/**
 * Example function to trigger music/SFX generation
 * @param {Array} sceneChanges - Array of scene change timestamps
 * @param {Array} detectedImpacts - Array of impact event objects
 */

allCues = []



function triggerCueingSystem(sceneChanges, detectedImpacts, audioFilesFromBackend = []) {



    sceneChanges.forEach(time => {
        console.log(`Trigger scene change cue at ${time} seconds.`);
        const cue = renderCueArrow(time, 'Scene Change', 'scene');
        allCues.push(cue);
    });

    // detectedImpacts.forEach(event => {
        // console.log(`Trigger impact cue at ${event.timestamp} seconds for ${event.object}.`);
    // });

    const audioFiles = (audioFilesFromBackend.length > 0)
        ? audioFilesFromBackend
        : ['Galaga.mp3', 'spacescore.mp3', 'spacescore.mp3']; // fallback for testing

    const sceneTiming = (syncToSceneChanges && sceneChanges?.length)
        ? sceneChanges
        : audioFiles.map((_, i) => i * 2); // fallback timing if scene changes are missing

    if (newbatch == true){
        createDownloadLinks(audioFiles, sceneTiming);
        console.log('asouhfsaou')
    }
}






const timelineBar = document.getElementById('timeline-bar');

if (window.location.pathname == '/doseedo1.html') {


timelineBar.addEventListener('click', function (e) {
    const rect = timelineBar.getBoundingClientRect();
    const clickX = e.clientX - rect.left;

    const containerWidth = document.getElementById('download-links').offsetWidth * currentZoomLevel;
    
    const pixelsPerSecond = containerWidth / originalLengthInSeconds;

    const timeClicked = clickX / pixelsPerSecond;

    videoPreviewsrc.currentTime = Math.min(Math.max(timeClicked, 0), originalLengthInSeconds);
});






const selectionBox = document.getElementById('selection-box');
const downloadBox = document.querySelector('.downloadbox');

let startX = 0;
let startY = 0;

// downloadBox.addEventListener('mousedown', (e) => {
//     if (e.button !== 0) return; // Only left click

//     const rect = downloadBox.getBoundingClientRect();
//     startX = e.clientX - rect.left;
//     startY = e.clientY - rect.top;

//     selectionBox.style.left = `${startX}px`;
//     selectionBox.style.top = `${startY}px`;
//     selectionBox.style.width = `0px`;
//     selectionBox.style.height = `0px`;
//     selectionBox.style.display = 'block';

//     function onMouseMove(eMove) {
//         const currentX = eMove.clientX - rect.left;
//         const currentY = eMove.clientY - rect.top;

//         const x = Math.min(currentX, startX);
//         const y = Math.min(currentY, startY);
//         const width = Math.abs(currentX - startX);
//         const height = Math.abs(currentY - startY);

//         selectionBox.style.left = `${x}px`;
//         selectionBox.style.top = `${y}px`;
//         selectionBox.style.width = `${width}px`;
//         selectionBox.style.height = `${height}px`;
//     }

//     function onMouseUp(eUp) {
//         document.removeEventListener('mousemove', onMouseMove);
//         document.removeEventListener('mouseup', onMouseUp);
//         selectionBox.style.display = 'none';

//         const selectionRect = selectionBox.getBoundingClientRect();

//         document.querySelectorAll('#download-links li').forEach(track => {
//             const trackRect = track.getBoundingClientRect();
//             const intersects = !(
//                 trackRect.right < selectionRect.left ||
//                 trackRect.left > selectionRect.right ||
//                 trackRect.bottom < selectionRect.top ||
//                 trackRect.top > selectionRect.bottom
//             );

//             if (intersects) {
//                 if (!selectedTracks.includes(track)) {
//                     track.classList.add('selected');
//                     selectedTracks.push(track);
//                 }
//             }
//         });
//     }

//     document.addEventListener('mousemove', onMouseMove);
//     document.addEventListener('mouseup', onMouseUp);
// });




const cursorLine = document.getElementById('cursor-line');
const cursorTime = document.getElementById('cursor-time');

document.querySelector('.downloadbox').addEventListener('mousemove', (e) => {
    const container = document.querySelector('.downloadbox');
    const bounds = container.getBoundingClientRect();
    const x = e.clientX - bounds.left;

    const visibleWidth = container.offsetWidth * currentZoomLevel;
    const seconds = (x / visibleWidth) * originalLengthInSeconds;

    // cursorLine.style.left = `${x}px`;
    cursorTime.innerText = seconds.toFixed(2) + 's';
    // cursorTime.style.left = `${x + 5}px`;
    // cursorLine.style.display = 'block';
});


document.querySelector('.downloadbox').addEventListener('mouseleave', () => {
    cursorLine.style.display = 'none';
});


}



let newbatch = false


function pollTaskStatus(taskId, callback) {
    fetch(`https://doseedo.com/task/${taskId}`)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'completed') {
                // Prepend the backend domain to audio paths
                const fullPaths = data.result.map(path => `https://doseedo.com${path}`);

                if (!globalSceneChanges || !globalSceneChanges.length) {
                    console.warn("globalSceneChanges not ready, falling back to dummy timings");
                    globalSceneChanges = fullPaths.map((_, i) => i * 2);
                }
                // Call your callback if needed
                callback(fullPaths);

                // Start UI logic
                newbatch = true
                createDownloadLinks(fullPaths, globalSceneChanges);

                updateStatus('Task completed. Download available.');
                document.querySelector('#genmus').style.pointerEvents = 'auto';
                document.querySelector('#genmus').style.opacity = '1';
                document.querySelector('#gensfx').style.opacity = '1';
                document.querySelector('#gensfx').style.pointerEvents = 'auto';
                loading.style.visibility = 'hidden';
            } else {
                updateStatus(`Processing... (${data.status})`);
                setTimeout(() => pollTaskStatus(taskId, callback), 2000);
            }

                    // 🧠 Ensure this exists and is in scope!


        })
        .catch(error => {
            console.error('Error polling task status:', error);
            updateStatus('Error during task status polling.');
            loading.style.visibility = 'hidden';
        });
}




let selectedTracks = [];
let trackStateMap = new Map(); // map of original positions




function setupGlobalControls() {

    const downloadsContainer = document.querySelector('.downloadbox');
    const footer = document.querySelector('#footer');

    // Prevent duplicate creation
    if (document.getElementById('slider-container')) return;


 

    const sliderContainer = document.createElement("div")
    sliderContainer.style.marginBottom = "10px";
    // sliderContainer.style.display = "flex";
    sliderContainer.style.alignItems = "center";
    sliderContainer.style.gap = "10px";
    sliderContainer.id = "slider-container";

    const heightSlider = document.createElement("input");
    heightSlider.type = "range";
    heightSlider.min = "30";
    heightSlider.max = "150";
    heightSlider.value = "60";
    heightSlider.id = "height-slider"; // assign ID to access later

    let zoomSlider = document.createElement("input");
    zoomSlider.type = "range";
    zoomSlider.min = "1";
    zoomSlider.max = "10";
    zoomSlider.step = "0.1";
    zoomSlider.value = "1";
    zoomSlider.id = "zoom-slider"; // assign ID to access later

    


    const deleteButton = document.createElement("button");
    const syncbutton = document.createElement("button");

// Add Font Awesome icon
    const icon = document.createElement("i");
    icon.className = "fa-solid fa-delete-left";

    const icon2 = document.createElement("i");
    icon2.className = "fa-solid fa-sync";

    // Add text node
    // const buttonText = document.createTextNode("");

    // Append icon and text to button
    deleteButton.appendChild(icon);
    deleteButton.classList.add("delete-track-btn");
    // deleteButton.style.margin = "10px";
    deleteButton.id = "delete-track-btn";

    syncbutton.appendChild(icon2);
    syncbutton.classList.add("sync-track-btn");
    // syncbutton.style.margin = "10px";
    syncbutton.id = "sync-track-btn";


    sliderContainer.appendChild(heightSlider);
    sliderContainer.appendChild(zoomSlider);

    sliderContainer.appendChild(deleteButton);
    sliderContainer.appendChild(syncbutton);


        // insert it right AFTER timeline-wrapper

        const timelineWrapper = document.querySelector('.timeline-wrapper');


    // timelineWrapper.insertAdjacentElement('beforebegin', sliderContainer);
    footer.appendChild(sliderContainer);
    
}






function createDownloadLinks(filePaths, sceneChanges, containerId = 'download-links') {
    let downloadList = document.getElementById('download-links');
    downloadList = document.getElementById(containerId);
    if (!downloadList) {
        console.error(`❌ Container with ID '${containerId}' not found.`);
        return;
    }

    if (!Array.isArray(sceneChanges) || sceneChanges.length === 0) {
        console.warn("⚠️ sceneChanges not ready or empty. Using fallback timings.");
        sceneChanges = filePaths.map((_, i) => i * 2); // fallback scene timing
    }

    let selectedTracks = [];
    const trackStateMap = new Map();
    sceneChanges = globalSceneChanges

    function getFileName(path) {
        if (typeof path === 'string') {
            return path.split('/').pop().replace(/\.[^/.]+$/, "");
        } else if (path instanceof File) {
            return path.name.replace(/\.[^/.]+$/, ""); // fallback for File objects
        }
        return 'Unknown';
    }
    

    // let downloadList = document.getElementById('download-links');
    // downloadList.classList.add('expanded');
    if (window.location.pathname === '/home.html') {
        downloadList = document.querySelector('.homelinks');
    }

    if (!window.trackAppendMode) {
        downloadList.innerHTML = "";
    }
    


    const downloadsContainer = document.querySelector('.downloadbox');

    const heightSlider = document.getElementById("height-slider");
    let zoomSlider = document.getElementById("zoom-slider");
    const deleteButton = document.getElementById("delete-track-btn");
    const syncbutton = document.getElementById("sync-track-btn");
    
    const tracklistContainer = document.querySelector('.tracklist');
    if (!window.trackAppendMode) {
        tracklistContainer.innerHTML = "";  // Clear existing labels
    }

  

    zoomSlider.addEventListener('input', () => {
        currentZoomLevel = parseFloat(zoomSlider.value);
        const containerWidth = document.getElementById('download-links').offsetWidth;
        // const containerWidth = 1000; // Fixed base width
        renderTimeline(originalLengthInSeconds, currentZoomLevel, containerWidth);
        document.querySelectorAll('#download-links li, #download-links2 li, #download-links3 li').forEach(updateTrackPosition);
    });
;




    


    deleteButton.addEventListener("click", () => {
        if (selectedTracks.length) {
            selectedTracks.forEach(track => track.remove());
            selectedTracks = [];
        } else {
            alert("No track selected to delete!");
        }
    });



    function updateTrackPosition(track) {
        if (!track || !track._meta) return;
        const meta = track._meta;
        const container = track.querySelector('.waveform-container');
        const containerWidth = downloadList.offsetWidth * currentZoomLevel;
        const pixelsPerSecond = containerWidth / originalLengthInSeconds;
        const translateX = meta.startPositionInSeconds * pixelsPerSecond;
        const pixelWidth = meta.trackDuration * pixelsPerSecond;

        container.style.left = `${translateX}px`;
        container.style.width = `${pixelWidth}px`;

        console.log('updated')

        const wavesurfer = track._wavesurfer;
        if (wavesurfer) {
            const fullDuration = wavesurfer.getDuration();
            const trimStart = Math.max(0, meta.trimStart || 0);
            const trimEnd = Math.min(fullDuration, (meta.trimEnd !== undefined ? meta.trimEnd : fullDuration));
            const trimDuration = trimEnd - trimStart;
            if (!isNaN(trimStart) && !isNaN(trimEnd) && trimDuration > 0) {
                wavesurfer.setOptions({ minPxPerSec: pixelWidth / trimDuration });
                wavesurfer.seekTo(trimStart / fullDuration);
            }
        }

        // console.log(`Track: Start = ${meta.startPositionInSeconds.toFixed(2)}s | Duration = ${meta.trackDuration.toFixed(2)}s | End = ${meta.endPositionInSeconds.toFixed(2)}s`);
    }

    downloadList.addEventListener('wheel', (e) => {
        const zoomSlider = document.getElementById('zoom-slider');
    
        if (!e.ctrlKey) return; // optional: only zoom when holding Ctrl
        e.preventDefault();
    
        const delta = e.deltaY;
        const zoomStep = 0.1;
    
        // Update zoom level
            // Invert the scroll logic for natural feel

        if (delta < 0) {
            currentZoomLevel = Math.min(10, currentZoomLevel + zoomStep); // zoom in
        } else {
            currentZoomLevel = Math.max(1, currentZoomLevel - zoomStep); // zoom out
        }
   
        // Sync zoom slider
        zoomSlider.value = currentZoomLevel.toFixed(2);
    
        // Apply zoom effect
        const containerWidth = downloadList.offsetWidth;
        renderTimeline(originalLengthInSeconds, currentZoomLevel, containerWidth);
        document.querySelectorAll('#download-links li, #download-links2 li, #download-links3 li').forEach(updateTrackPosition);
    });


    let currentLiCount = downloadList.querySelectorAll('li').length;

    // document.querySelector('.downloadbox').style.left = '10%'

    filePaths.forEach((filePath, index) => {
        

        const audioLink = filePath;



        const listItem = document.createElement("li");
        
        // listItem.style.position = 'absolute';
        listItem.dataset.originalIndex = currentLiCount;
        // ✅ Calculate position based on max .top from existing tracks
        let maxTop = 0;
        downloadList.querySelectorAll('li').forEach(li => {
        const t = parseInt(li.style.top || '0', 10);
        if (!isNaN(t) && t >= maxTop) {
            maxTop = t;
        }
        });

        const newTop = currentLiCount * 60;
        listItem.style.top = `${newTop}px`;
        listItem.dataset.originalTop = newTop;
        listItem.dataset.originalIndex = currentLiCount;
        currentLiCount++; // increment after use

        listItem.style.cursor = 'pointer';
        listItem.style.width = '100%';
        listItem.style.display = 'flex';
        listItem.style.alignItems = 'center';
        // listItem.dataset.originalIndex = index;

        const labelContainer = document.createElement("div");
        labelContainer.style.position = 'relative';
        labelContainer.style.zIndex = '1';
        labelContainer.style.marginTop = '10px';

        const trackLabel = document.createElement("div");
        const isServerAudio = typeof filePath === 'string' && filePath.includes('doseedo.com/media/audio');
        const labelName = isServerAudio ? 'Original Audio' : getFileName(filePath);
        trackLabel.innerText = `${index + 1}: ${labelName}`;


        trackLabel.style.color = "white";
        // trackLabel.style.paddingRight = "10px";


        const playButton = document.createElement("button");
        playButton.innerText = "⏯";
        playButton.style.marginRight = "10px";
        playButton.style.cursor = "pointer";

        const muteButton = document.createElement("button");
        muteButton.innerText = "M";
        muteButton.style.marginRight = "10px";
        muteButton.style.cursor = "pointer";


        const soloButton = document.createElement("button");
        soloButton.innerText = "S";
        soloButton.style.marginRight = "10px";
        soloButton.style.cursor = "pointer";

        labelContainer.appendChild(trackLabel);
        labelContainer.appendChild(playButton);
        playButton.style.visibility = 'hidden'
        labelContainer.appendChild(muteButton);
        labelContainer.appendChild(soloButton);
        labelContainer.id = "Tracklabels"


          
        // listItem.appendChild(labelContainer);


        // Map download-list ID to its tracklist
        let tracklistForThisDownloadList;

        if (downloadList.id === 'download-links') {
          tracklistForThisDownloadList = document.querySelector('#muss .tracklist');
        } else if (downloadList.id === 'download-links2') {
          tracklistForThisDownloadList = document.querySelector('#sfxs .tracklist');
        } else if (downloadList.id === 'download-links3') {
          tracklistForThisDownloadList = document.querySelector('#vos .tracklist');
        }
        
        if (tracklistForThisDownloadList) {
          tracklistForThisDownloadList.appendChild(labelContainer);
        //   tracklistForThisDownloadList.classList.add('expanded');
        } else {
          console.warn(`⚠️ No matching tracklist found for container ID: ${downloadList.id}`);
        }
        



        const waveformContainer = document.createElement("div");
        waveformContainer.className = 'waveform-container';
        waveformContainer.style.position = 'absolute';
        waveformContainer.style.height = '50px';
        waveformContainer.style.border = '1px solid #000';
        // waveformContainer.style.background = 'dark grey';


        let baseHue;
        if (downloadList.id === 'download-links') {
        baseHue = 230; // Music
        } else if (downloadList.id === 'download-links2') {
        baseHue = 180; // SFX
        } else if (downloadList.id === 'download-links3') {
        baseHue = 300; // VO
        } else {
        baseHue = 200; // Default fallback
        }

        const hue = baseHue + (currentLiCount * 10) % 70;
        const pastelColor = `hsl(${hue}, 30%, 65%)`;
        waveformContainer.style.background = pastelColor;

        listItem.appendChild(waveformContainer);


        document.querySelectorAll('#Tracklabels button').forEach(button => {
            button.addEventListener('click', () => {
              button.classList.toggle('selected');
            });
          });
        
        
        // ✅ Ensure new track shows even if container is collapsed


        // const downloadList = document.getElementById('download-links');
        // const downloadList = document.getElementById('download-links');
        const isCollapsed = !downloadList.classList.contains('expanded');
        
        // ✅ Temporarily expand the container so new track will be visible

          
        // Append the new track
        downloadList.appendChild(listItem);
        
        
        // ✅ Recalculate tops so everything is spaced properly
        const allTrackItems = document.querySelectorAll('#download-links > li');
        allTrackItems.forEach((li, i) => {
          const top = i * 60;
          li.style.top = `${top}px`;
          li.dataset.originalTop = top;
          const zind = i * 100;
          li.style.zIndex = `${zind}`;
        });

        
        // ✅ Return to collapsed state if it started that way
        if (isCollapsed) {
          setTimeout(() => {
            downloadList.classList.remove('expanded');
            // Flatten all tops after collapse
            allTrackItems.forEach(li => {
              li.style.top = '0px';
            });
          }, 50);
        }
        



        const rand = Math.floor(Math.random() * 50);
        const wavesurfer = WaveSurfer.create({
            backend: 'WebAudio',
            container: waveformContainer,
            waveColor: `white`,
            progressColor: 'white',

            barWidth: 2,
            height: parseInt(heightSlider.value),
            normalize: true
        });

    
        if (audioLink instanceof File) {
            wavesurfer.loadBlob(audioLink);
            console.log("🎧 Loading local file blob into WaveSurfer:", audioLink);
        } else {
            console.log("🎧 Fetching and loading remote audio into WaveSurfer:", audioLink);
            wavesurfer.load(audioLink);
        }
        
        



        listItem._wavesurfer = wavesurfer;



        playButton.addEventListener("click", () => {
            const meta = listItem._meta;
            const fullDuration = wavesurfer.getDuration();
        
            if (!wavesurfer.isPlaying()) {
                // Clamp trimStart to fullDuration range
                if (!meta || typeof meta.trimStart !== 'number') return;

                wavesurfer.seekTo(start / fullDuration);
        
                // Playback limiter
                let frameId;
                const stopAtTrimEnd = () => {
                    const current = wavesurfer.getCurrentTime();
                    if (current >= meta.trimEnd) {
                        wavesurfer.pause();
                        wavesurfer.seekTo(start / fullDuration);
                        playButton.innerText = "▶";
                        cancelAnimationFrame(frameId);
                    } else {
                        frameId = requestAnimationFrame(stopAtTrimEnd);
                    }
                };
        
                frameId = requestAnimationFrame(stopAtTrimEnd);
                wavesurfer.play();
                playButton.innerText = "⏸";
            } else {
                wavesurfer.pause();
                playButton.innerText = "▶";
            }
        });
        
        let startPositionInSeconds = sceneChanges[index] ?? index * 2;  // fallback if undefined
        let nextStart = sceneChanges[index + 1];
        let trackDuration = nextStart !== undefined
            ? nextStart - startPositionInSeconds
            : null; // Defer until WaveSurfer is ready

        trackDuration = Math.max(0.1, trackDuration);  // Avoid zero-duration



        
        listItem._meta = {
            startPositionInSeconds,
            trackDuration,
            trimStart: 0,
            trimEnd: startPositionInSeconds + trackDuration,
            get endPositionInSeconds() {
                return this.startPositionInSeconds + this.trackDuration;
            }
        };

        
        wavesurfer.on('ready', () => {

            console.log("✅ WaveSurfer is ready");
            console.log("⏱ Full Duration:", wavesurfer.getDuration());
        
            wavesurfer.isReady = true;
            const meta = listItem._meta;
            const fullDuration = wavesurfer.getDuration();
        
            // 🔥 Fix: Set trackDuration to actual audio length immediately
       
            // 🔥 Fix: Force full duration for extracted/original audio
            if (listItem.dataset.trackName === 'Original Audio' || isServerAudio) {
                meta.trackDuration = fullDuration;
                meta.trimStart = 0;
                meta.trimEnd = fullDuration;
            } else {
                meta.trackDuration = nextStart !== undefined
                    ? Math.min(fullDuration, nextStart - meta.startPositionInSeconds)
                    : fullDuration;
                meta.trimStart = Math.max(0, meta.trimStart ?? meta.startPositionInSeconds);
                meta.trimEnd = Math.min(fullDuration, meta.trimStart + meta.trackDuration);
            }

       
            const visibleDuration = meta.trimEnd - meta.trimStart;
        
            // 👇 Then calculate pixel size for that trimmed/actual audio
            const containerWidth = downloadList.offsetWidth * currentZoomLevel;
            const pixelsPerSecond = containerWidth / originalLengthInSeconds;
            const pixelWidth = visibleDuration * pixelsPerSecond;
        
            // 👇 Apply accurate display settings
            wavesurfer.setOptions({
                minPxPerSec: pixelWidth / visibleDuration,
                autoCenter: false,
                interact: false,
                hideScrollbar: true
            });
        
            // 👇 Force visual correction (if needed)
            if (wavesurfer.drawer?.container) {
                wavesurfer.drawer.container.style.width = `${pixelWidth}px`;
                wavesurfer.drawer.progress(0);
            }
        
            // 👇 Zoom update (preserves waveform scaling)
            wavesurfer.zoom(pixelWidth / visibleDuration);
        
            // 👇 Ensure scroll and visual sync
            if (wavesurfer.drawer?.wrapper) {
                wavesurfer.drawer.wrapper.scrollLeft = 0;
            }
        
            // 👇 Defer final updates to next frame
            requestAnimationFrame(() => {
                if (wavesurfer.drawer?.container) {
                    wavesurfer.drawer.container.style.width = `${pixelWidth}px`;
                    wavesurfer.drawer.progress(0);
                }
                updateTrackPosition(listItem); // force snap to timeline
            });
        });
        
        
        
        
        

        interact(waveformContainer).resizable({
            edges: { left: true, right: true },
            listeners: {
                start(event) {
                    if (!selectedTracks.includes(listItem)) {
                        selectedTracks.forEach(el => el.classList.remove('selected'));
                        selectedTracks = [listItem];
                        listItem.classList.add('selected');
                    }
                },
                move(event) {
                    const parentWidth = downloadList.offsetWidth * currentZoomLevel;
                    const pixelsPerSecond = parentWidth / originalLengthInSeconds;

                    const deltaStartTime = event.deltaRect.left / pixelsPerSecond;
                    const deltaEndTime = event.deltaRect.width / pixelsPerSecond;

                    selectedTracks.forEach(track => {
                        let meta = track._meta;
                        const wavesurfer = track._wavesurfer;
                        const fullDuration = wavesurfer.getDuration();

                        if (event.edges.left) {
                            let newStart = Math.max(0, meta.startPositionInSeconds + deltaStartTime);
                            let durationDelta = meta.startPositionInSeconds - newStart;
                            meta.startPositionInSeconds = newStart;
                            meta.trackDuration = Math.min(fullDuration, Math.max(0.1, meta.trackDuration + durationDelta));
                        }

                        if (event.edges.right) {
                            meta.trackDuration = Math.min(fullDuration, Math.max(0.1, meta.trackDuration + deltaEndTime));
                        }

                        meta.trimStart = 0;
                        meta.trimEnd = Math.min(fullDuration, meta.trackDuration);
                        updateTrackPosition(track);
                    });
                }
            }
        });

        interact(waveformContainer).draggable({
            axis: 'x',
            listeners: {
                start(event) {
                    if (!selectedTracks.includes(listItem)) {
                        selectedTracks.forEach(el => {
                            el.querySelector('.waveform-container')?.classList.remove('selected');
                        });
                        selectedTracks = [listItem];
                        waveformContainer.classList.add('selected');
                    }
                },
                move(event) {
                    const parentWidth = downloadList.offsetWidth * currentZoomLevel;
                    const pixelsPerSecond = parentWidth / originalLengthInSeconds;
                    const deltaX = event.dx;
        
                    selectedTracks.forEach(track => {
                        let meta = track._meta;
                        const currentLeft = meta.startPositionInSeconds * pixelsPerSecond;
                        const newLeft = Math.max(0, currentLeft + deltaX);
                        meta.startPositionInSeconds = newLeft / pixelsPerSecond;
                        updateTrackPosition(track);
                    });
                }
            }
        });
        

        waveformContainer.addEventListener('click', (e) => {
            const isShift = e.shiftKey;
        
            if (isShift) {
                if (selectedTracks.includes(listItem)) {
                    waveformContainer.classList.remove('selected');
                    selectedTracks = selectedTracks.filter(el => el !== listItem);
                } else {
                    waveformContainer.classList.add('selected');
                    selectedTracks.push(listItem);
                }
            } else {
                // Deselect previously selected tracks visually and logically
                selectedTracks.forEach(el => {
                    el.querySelector('.waveform-container')?.classList.remove('selected');
                });
                selectedTracks = [];
        
                waveformContainer.classList.add('selected');
                selectedTracks.push(listItem);
            }
        
            e.stopPropagation();
        });

        syncbutton.addEventListener("click", () => {
            newbatch = false;
        
            if (selectedTracks.length !== 1) {
                console.log("Please select exactly one track to sync.");
                return;
            }
        
            const track = selectedTracks[0];
            const originalIndex = parseInt(track.dataset.originalIndex, 10);
        
            if (isNaN(originalIndex)) {
                console.warn("Track missing original index.");
                return;
            }
        
            const meta = track._meta;
            meta.startPositionInSeconds = globalSceneChanges[originalIndex];
        
            console.log(`Synced to scene change ${originalIndex}:`, globalSceneChanges[originalIndex]);
        
            updateTrackPosition(track);
        });
        
        
    });

    setTimeout(() => {
        document.querySelectorAll('#download-links li, #download-links2 li, #download-links3 li').forEach(updateTrackPosition);
    }, 0);

    document.addEventListener('click', (e) => {
        // Deselect only if the click was not on any .waveform-container
        if (!e.target.closest('.waveform-container')) {
            selectedTracks.forEach(el => {
                el.querySelector('.waveform-container')?.classList.remove('selected');
            });
            selectedTracks = [];
        }
    });

    newbatch = false
    

    const containerWidth = document.getElementById('download-links').offsetWidth;
    // const containerWidth = 1000; // Fixed base width
    renderTimeline(originalLengthInSeconds, currentZoomLevel, containerWidth);
}

let downloadList = document.getElementById('download-links');




function setupTrackDropZone(zoneId, targetDownloadListId) {
    const dropZone = document.getElementById(zoneId);

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.backgroundColor = 'rgba(255,255,255,0.05)';
        dropZone.style.border = '2px dashed #aaa';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.backgroundColor = '';
        dropZone.style.border = '';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.backgroundColor = '';
        dropZone.style.border = '';

        const audioFiles = Array.from(e.dataTransfer.files).filter(file =>
            file.type.startsWith("audio/")
        );

        if (audioFiles.length === 0) {
            alert('Please drop valid .mp3 or .wav audio files.');
            return;
        }

        window.trackAppendMode = true;
        createDownloadLinks(audioFiles, sceneChanges, targetDownloadListId);
        window.trackAppendMode = false;
    });
}

  





let cursor = document.getElementById('timeline-cursor');
let cursor2 = document.getElementById('timeline-cursor2');





function updateTimelineCursor() {
    const videoTime = videoPreviewsrc.currentTime;
    const containerWidth = document.getElementById('download-links').offsetWidth * currentZoomLevel;
    const pixelsPerSecond = containerWidth / originalLengthInSeconds;

    const cursorX = videoTime * pixelsPerSecond;
    cursor.style.left = `${cursorX}px`;
    cursor2.style.left = `${cursorX}px`;

    requestAnimationFrame(updateTimelineCursor);
}







function renderTimeline(videoDuration, zoomLevel, containerWidth) {
    const timeline = document.getElementById('timeline-bar');

        // Save the cursor if it exists



    timeline.innerHTML = '';

    cursor = document.getElementById('timeline-cursor');
    cursor2 = document.getElementById('timeline-cursor2');
    
// Create cursors if missing
if (!cursor) {
    cursor = document.createElement('div');
    cursor.id = 'timeline-cursor';
    cursor.style.position = 'absolute';
    cursor.style.top = '30px';
    cursor.style.width = '1px';
    cursor.style.height = '10000%';
    cursor.style.background = 'off-white';
    cursor.style.zIndex = '10';
    cursor.style.pointerEvents = 'none';
    timeline.appendChild(cursor);
}

if (!cursor2) {

    cursor2 = document.createElement('div');
    cursor2.id = 'timeline-cursor2';
    cursor2.style.position = 'absolute';
    cursor2.style.top = '0';
    cursor2.style.left = '0';
    cursor2.style.width = '1px';
    cursor2.style.height = '0';
    cursor2.style.borderLeft = '10px solid transparent';
    cursor2.style.borderRight = '10px solid transparent';
    cursor2.style.borderTop = '15px solid white'; // arrow color
    cursor2.style.zIndex = '10';
    cursor2.style.pointerEvents = 'none';
    cursor2.style.transform = 'translateX(-50%)';
    timeline.appendChild(cursor2);

}



    const zoomedWidth = containerWidth * zoomLevel;
    timeline.style.width = `${zoomedWidth}px`;

    const pixelsPerSecond = zoomedWidth / videoDuration;
    const secondInterval = 1; // major ticks
    const subTickInterval = 0.1; // 100ms minor ticks


    // Draw minor (millisecond) ticks
if (showMillisecondTicks) {
    for (let t = 0; t <= videoDuration; t += subTickInterval) {
        const tick = document.createElement('div');
        tick.className = 'tick minor';
        tick.style.position = 'absolute';
        tick.style.left = `${t * pixelsPerSecond}px`;
        tick.style.height = '50%';
        tick.style.width = '1px';
        tick.style.backgroundColor = '#666';
        timeline.appendChild(tick);
    }
}

    // Draw major (1s) ticks + labels
    for (let t = 0; t <= videoDuration; t += secondInterval) {
        const tick = document.createElement('div');
        tick.className = 'tick major';
        tick.style.position = 'absolute';
        tick.style.left = `${t * pixelsPerSecond}px`;
        tick.style.height = '100%';
        tick.style.width = '1px';
        // tick.style.backgroundColor = '#ccc';
        timeline.appendChild(tick);

        const label = document.createElement('div');
        label.className = 'tick-label';
        label.innerText = `${t.toFixed(0)}s`;
        label.style.position = 'absolute';
        label.style.top = '10px';
        label.style.left = `${t * pixelsPerSecond + 3}px`;
        label.style.fontSize = '10px';
        label.style.color = '#ccc';
        timeline.appendChild(label);
    }

    allCues.forEach(cue => {
        cue.style.left = `${cue.dataset.time * pixelsPerSecond}px`;
        timeline.appendChild(cue);
    });
}

function renderCueArrow(time, label, type) {
    const arrow = document.createElement('div');
    arrow.className = 'cue-arrow';
    arrow.dataset.time = time; // <-- store the time for later repositioning
    arrow.dataset.type = type;
    arrow.title = label;
    return arrow;
}






function exitvid (){
    document.querySelector('.videoprevsrc').style.display = 'none';
    document.querySelector('.videoprevsrc').src = '';
    document.querySelector('#switchbtn').style.display = 'block';
    document.querySelector('.glow-container').style.display = 'block';
    document.querySelector('#vidx').style.display = 'none';

    // 🛠 Reset file input so same file can be selected again
    document.getElementById('videoFile').value = '';
}


// Video input and preview handling
const videoInput = document.getElementById('videoFile');
let videoPreview = document.getElementById('videoPreview');
const videoPreviewsrc = document.querySelector('.videoprevsrc');
let customStartTime = null;






// Only clear hasPlayed flags and pause all on seek
// When user seeks: jump to right position and pause or play if needed






function syncWaveSurfersToVideo() {
    const currentTime = videoPreviewsrc.currentTime;

    document.querySelectorAll('#download-links li, #download-links2 li, #download-links3 li').forEach((track, i) => {
        const wavesurfer = track._wavesurfer;
        const meta = track._meta;

        if (!wavesurfer || !meta || !wavesurfer.isReady) {
            console.warn(`Track ${i} skipped — not ready`);
            return;
        }

        const {
            startPositionInSeconds,
            endPositionInSeconds,
            trimStart = 0,
            trimEnd = wavesurfer.getDuration()
        } = meta;

        const isInWindow = currentTime >= startPositionInSeconds && currentTime <= endPositionInSeconds;

        if (isInWindow) {
            const offset = currentTime - startPositionInSeconds;
            const targetAudioTime = Math.min(trimStart + offset, trimEnd);

            console.log(`🟢 Video inside Track ${i}: currentTime=${currentTime.toFixed(2)} → audioTime=${targetAudioTime.toFixed(2)}`);

            if (Math.abs(wavesurfer.getCurrentTime() - targetAudioTime) > 0.05) {
                console.log(`🔁 Seeking Track ${i} to ${targetAudioTime.toFixed(2)}`);
                wavesurfer.setTime(targetAudioTime);
            }

            if (!videoPreviewsrc.paused && !wavesurfer.isPlaying()) {
                console.log(`▶️ Playing Track ${i}`);
                wavesurfer.play().catch(e => {
                    console.error(`❌ Play failed on Track ${i}`, e);
                });
            }
            
        } else {
            if (wavesurfer.isPlaying()) {
                console.log(`⏸️ Pausing Track ${i}`);
                wavesurfer.pause();
            }
            wavesurfer.setTime(trimStart);
        }
    });
}
















if (window.location.pathname == '/doseedo1.html') {
    videoInput.addEventListener('change', function () {
        const file = videoInput.files[0];
        const vidx = document.getElementById('vidx');
        const switchbtn = document.getElementById('switchbtn');
    
        switchbtn.style.display = 'none';
        vidx.style.display = 'block';
    
        if (file) {
            

            const reader = new FileReader();
            reader.onload = function (e) {
                videoPreviewsrc.src = e.target.result;
                videoPreviewsrc.type = file.type;
    
                document.querySelector('.videoprevsrc').style.display = 'block';
                player.source = {
                    type: 'video',
                    sources: [{
                        src: e.target.result,
                        type: file.type
                    }]
                };
    
                // ✅ Wait for metadata before proceeding
               
                videoPreviewsrc.onloadedmetadata = function () {
                    originalLengthInSeconds = videoPreviewsrc.duration;
                    // const containerWidth = 1000; // Fixed base width
        
                    const containerWidth = document.getElementById('download-links').offsetWidth;
                   // Calculate zoom level to fit entire video in view
                    const fitToScreenZoomLevel = containerWidth / originalLengthInSeconds;
                    currentZoomLevel = fitToScreenZoomLevel;
                
                    let zoomSlider = document.getElementById("zoom-slider");

                    zoomSlider.min = 1; // or 0.1 if you want more zoom-in range
                    zoomSlider.max = 10;
                    zoomSlider.step = 0.1;
                    zoomSlider.value = fitToScreenZoomLevel;
                
                   renderTimeline(originalLengthInSeconds, currentZoomLevel, containerWidth);
                
                    sceneChanges = [0, 2, 4];
                    // const audioFiles = ['Galaga.mp3', 'Galaga.mp3', 'Galaga.mp3'];
                    // createDownloadLinks(audioFiles, sceneChanges);
                
                    player.play();
                };
                
                
            };
    
            reader.readAsDataURL(file);
        } else {
            videoPreviewsrc.src = '';
        }
    });
    


videoPreviewsrc.addEventListener('play', function() {



    console.log('Video is now playing.');
    // You can call your desired function here
    requestAnimationFrame(updateTimelineCursor);

    waitForButtonsThenClick(); // 👈 new
    

    
});


    


videoPreviewsrc.addEventListener('pause', function() {
    console.log('Video is now playing.');
    // You can call your desired function here
    clickButtons();

    document.querySelectorAll('#download-links li, #download-links2 li, #download-links3 li').forEach(track => {
        const wavesurfer = track._wavesurfer;
        if (wavesurfer?.isPlaying()) {
            wavesurfer.pause();
        }
    });

});


document.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
        const tag = document.activeElement.tagName;

        // Don't trigger if user is typing
        if (!['INPUT', 'TEXTAREA'].includes(tag)) {
            e.preventDefault(); // Block default scroll behavior

            // Force blur if video is focused
            if (document.activeElement === videoPreviewsrc) {
                videoPreviewsrc.blur();
            }

            // Toggle play/pause
            if (videoPreviewsrc.paused) {
                videoPreviewsrc.play();
            } else {
                videoPreviewsrc.pause();
            }
        }
    }
});


videoPreviewsrc.addEventListener('click', (e) => {
    // Allow click behavior (like seeking), but prevent focus from sticking
    setTimeout(() => {
        if (document.activeElement === videoPreviewsrc) {
            videoPreviewsrc.blur();
        }
    }, 0);
});



}
let buttonIndices = [];

function addButtonIndex(index, element) {
    let doubledIndex = index * 2; // Assuming you still want to push index*2 to the array
    let indexInArray = buttonIndices.indexOf(doubledIndex);

    if (indexInArray === -1) {
        // Index not found in array, add it
        buttonIndices.push(doubledIndex);
        element.style.background = 'hotpink';
    } else {
        // Index found, remove it and reset the background
        buttonIndices.splice(indexInArray, 1);
        element.style.background = ''; // Reset to original background
    }

    console.log(buttonIndices);
}





function waitForButtonsThenClick() {
    const buttonsExist = document.querySelectorAll(
      '#download-links button, #download-links2 button, #download-links3 button'
    ).length > 0;
  
    if (buttonsExist) {
      clickButtons();
    } else {
    //   console.log("⏳ Waiting for buttons to appear...");
    //   setTimeout(waitForButtonsThenClick, 500);
    }
  }


function clickButtons() {
    const containerIds = ['download-links', 'download-links2', 'download-links3'];
  
    containerIds.forEach(containerId => {
      const container = document.getElementById(containerId);
      if (!container) {
        console.warn(`${containerId} not found.`);
        return;
      }
  
      const buttons = container.getElementsByTagName('button');
      if (buttons.length === 0) {
        console.warn(`No buttons found in ${containerId}.`);
        return;
      }
  
      buttonIndices.forEach(index => {
        if (index < buttons.length) {
          buttons[index].click();
          console.log(`✅ Clicked button at index ${index} in ${containerId}`);
        } else {
          console.warn(`❌ No button at index ${index} in ${containerId}`);
        }
      });
    });
  }
  








// Additional user registration and login functions
function registerUser(googleProfile) {
    let formData = new FormData();
    const username = document.getElementById('register-username').value;
    const email = document.getElementById('register-email').value;
    const password = document.getElementById('register-password').value;
  


    if (googleProfile) {
        // If registering through Google
        formData.append('username', googleProfile.getName());
        formData.append('email', googleProfile.getEmail());
        formData.append('password', googleProfile.getId()); // Optional, if you want to store Google ID
    } else {
        // If registering through traditional form
  
        formData.append('username', username);
        formData.append('email', email);
        formData.append('password', password);
    }

    fetch('https://doseedo.com/register/', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            document.querySelector('#signup-error-message').innerText = 'Username or email already exists.'
            document.querySelector('#signup-error-message').style.display = 'block'
            throw new Error('Invalid data.');

        }
        return response.json();
    })
    .then(data => {
        console.log('Registration success:', data);
        document.querySelector('#signup-error-message').innerText = 'Successfully registered'
        document.querySelector('#signup-error-message').style.color = 'white'
        isAuthenticated = true;
        using = email
        const subscriptionStatus = data.subscription ? "Pro+" : "Free";
        document.getElementById('user-subscription-status').textContent = subscriptionStatus;
        isPro = data.subscription ? true : false;

        localStorage.setItem('isAuth', true);
        localStorage.setItem('username', email);
        localStorage.setItem('ispro', subscriptionStatus);


        window.location.reload()
    })
    .catch(error => {
        console.error('Registration error:', error);
        // Error handling logic...
        document.querySelector('#signup-error-message').innerText = error
        document.querySelector('#signup-error-message').style.display = 'block'
    });
}


function loginUser(googleProfile) {
    let formData = new FormData();
    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;
    

    if (googleProfile) {
        // If logging in through Google
        formData.append('username', googleProfile.getEmail());
        formData.append('password', googleProfile.getId());
    } else {
        // If logging in through traditional form

        formData.append('username', username);
        formData.append('password', password);
        
    }

    fetch('https://doseedo.com/token/', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            document.querySelector('#signin-error-message').innerText = ' Incorrect username or password'
            document.querySelector('#signin-error-message').style.display = 'block'
            throw new Error('Incorrect username or password');

        }
        return response.json();
    })
    .then(data => {
        console.log('Login success:', data);
        isAuthenticated = true;


        toggleSignOutButton();
        // Display user information
        using = username
        console.log(using)
        document.getElementById('user-username').textContent = username;
        // Display subscription status
        
        const subscriptionStatus = data.subscription ? "Pro+" : "Free";
        document.getElementById('user-subscription-status').textContent = subscriptionStatus;
        isPro = data.subscription ? true : false;

        localStorage.setItem('isAuth', true);
        localStorage.setItem('username', username);
        localStorage.setItem('ispro', subscriptionStatus);
        
        // Other UI updates
        document.getElementById('user-info').style.display = 'block';
        document.getElementById('register-form').style.display = 'none';
        document.getElementById('login-form').style.display = 'none';
        document.getElementById('login-form').style.display = 'none';
        document.querySelector('.formdiv').style.display = 'none';
        // document.getElementById('signupbutton').style.display = 'none';
        document.getElementById('loginbutton').style.display = 'none';
        document.querySelector('.guest').style.display = 'none';
        location.reload();
    })
    .catch(error => {
        console.error('Login error:', error);
        document.querySelector('#signin-error-message').innerText = error
        document.querySelector('#signin-error-message').style.display = 'block'
        // Error handling logic...
    });
}


function signOut() {
    
    isAuthenticated = false;
    toggleSignOutButton();
    localStorage.removeItem('isAuth');
    localStorage.removeItem('username');
    localStorage.removeItem('ispro');
    window.location.reload();
    console.log()
    document.querySelector('.guest').style.display = 'block';


}

function toggleSignOutButton() {
    const signOutButton = document.getElementById("sign-out-button");
    if (isAuthenticated) {
        signOutButton.style.display = "inline";
    } else {

        signOutButton.style.display = "none";
        document.getElementById('user-info').style.display = '';
        document.getElementById('user-username').textContent = null;
        document.getElementById('user-info').style.display = 'none';
        // document.getElementById('signupbutton').style.display = 'inline';
        document.getElementById('loginbutton').style.display = 'inline';
    }
}

function postSubscriptionData(subscriptionID) {
    const formData = new FormData();

    console.log(using)
    formData.append('username', using);
    formData.append('subscription_id', subscriptionID);
    formData.append('email', using);

    fetch('https://doseedo.com/store-subscription/', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to store subscription information');
        }
        return response.json();
    })
    .then(data => {
        console.log('Subscription stored successfully:', data);
        localStorage.setItem('isAuth', true);
        localStorage.setItem('ispro', 'Pro+')
        isPro = true

        setTimeout(function() {
            location.reload();
        }, 1000); // Time in milliseconds (10000 milliseconds = 10 seconds)
        // You can add additional code here to handle successful subscription storage
    })
    .catch(error => console.error('Error storing subscription:', error));
}





function createNewSlider(audioLink) {
    // Assuming 'cues' is the container under the canvas where sliders should be added
    const cuesContainer = document.getElementsByClassName('cues')[0];
    const containerID = 'sliderContainer_' + audioLink.replace(/[^a-zA-Z0-9]/g, '_');

    // Create the main slider container
    const newSliderContainer = document.createElement('div');
    newSliderContainer.className = 'slider-container';
    newSliderContainer.id = `sliderContainer${sliderCount}`;
    cuesContainer.insertBefore(newSliderContainer, cuesContainer.firstChild); // Insert at the beginning

    // Create the slider
    const newSlider = document.createElement('div');
    newSlider.className = 'slider';
    newSlider.id = `slider${sliderCount}`;
    newSliderContainer.appendChild(newSlider);

    // Create time display element
    const timeDisplay = document.createElement('span');
    timeDisplay.className = 'time-display';
    timeDisplay.style.display = 'none'; // Initially hidden
    newSliderContainer.appendChild(timeDisplay);

    // Create fade settings input fields (initially hidden)
    const fadeInInput = document.createElement('input');
    fadeInInput.className = 'fade-input';
    fadeInInput.type = 'number';
    fadeInInput.placeholder = 'Fade In';
    fadeInInput.style.display = 'none';

    const fadeOutInput = document.createElement('input');
    fadeOutInput.className = 'fade-input';
    fadeOutInput.type = 'number';
    fadeOutInput.placeholder = 'Fade Out';
    fadeOutInput.style.display = 'none';

    const fadeInputsContainer = document.createElement('div');
    fadeInputsContainer.className = 'fade-inputs';
    fadeInputsContainer.appendChild(fadeInInput);
    fadeInputsContainer.appendChild(fadeOutInput);
    newSliderContainer.appendChild(fadeInputsContainer);
    newSliderContainer.id = audioLink

    // Create fade settings button
    const fadeSettingsButton = document.createElement('button');
    fadeSettingsButton.id = 'fadeButton';
    const icon = document.createElement('i');
    icon.classList.add("fa-solid", "fa-sliders");
    fadeSettingsButton.appendChild(icon);
    fadeSettingsButton.onclick = function() {
        // Toggle visibility of fade inputs
        fadeInInput.style.display = fadeInInput.style.display === 'block' ? 'none' : 'block';
        fadeOutInput.style.display = fadeOutInput.style.display === 'block' ? 'none' : 'block';
    };
    newSliderContainer.insertBefore(fadeSettingsButton, newSlider);

    // Create remove slider button
    const removeSliderButton = document.createElement('button');
    removeSliderButton.id = 'removeSlider';
    const removeIcon = document.createElement('i');
    removeIcon.classList.add("fa-solid", "fa-x");
    removeSliderButton.appendChild(removeIcon);
    removeSliderButton.onclick = function() {
        newSliderContainer.remove();
    };
    newSliderContainer.insertBefore(removeSliderButton, newSlider);



    // Initialize the slider (using noUiSlider)
    let moveTimeout;
    noUiSlider.create(newSlider, {
        start: [0, video.duration], // Example values, adjust as needed
        connect: true,
        range: {
            'min': 0,
            'max': video.duration // Example values, adjust as needed
        }
    });



    const audio = new Audio(audioLink);

// Load audio and set duration
audio.onloadedmetadata = () => {
    let audioDuration = audio.duration;

    // Slider events
    newSlider.noUiSlider.on('slide', function(values, handle) {
        let startValue = parseFloat(values[0]);
        let endValue = parseFloat(values[1]);
        let selectedDuration = endValue - startValue;

        // If the selected duration exceeds the audio duration
        if (selectedDuration > audioDuration) {
            if (handle === 0) { // Adjusting the start handle
                // Adjust the end handle to maintain the maximum allowed duration
                endValue = startValue + audioDuration;
                newSlider.noUiSlider.set([startValue, endValue]);
            } else { // Adjusting the end handle
                // Adjust the start handle to maintain the maximum allowed duration
                startValue = endValue - audioDuration;
                newSlider.noUiSlider.set([startValue, endValue]);
            }
        }

        clearTimeout(moveTimeout);
        updateCanvas(parseFloat(values[handle]));
        timeDisplay.textContent = `Start: ${startValue.toFixed(2)}s, End: ${endValue.toFixed(2)}s`;
        timeDisplay.style.display = 'block';

        moveTimeout = setTimeout(() => {
            timeDisplay.style.display = 'none';
        }, 1000);
    });

    newSlider.noUiSlider.on('change', function(values, handle) {
        // Update logic if needed...
    });
};

    
    

    sliderCount++;
}

function showcues(){

   document.querySelector('.buttons').style.display = 'none';
    document.querySelector('.cues').style.display = 'block';
}

function goback(){

    document.querySelector('.cues').style.display = 'none';
    document.querySelector('.buttons').style.display = 'block';
}




function checkTaskStatus(taskId) {
    fetch(`https://doseedo.com/task-status/${taskId}`)
        .then(response => response.json())
        .then(data => {
            console.log('Task status:', data);
            if (data.status === 'PENDING' || data.status === 'STARTED') {
                // If task is still running, check again after some time
                setTimeout(() => checkTaskStatus(taskId), 2000); // check every 2 seconds
            } else {
                // Task is complete or failed
                if (data.result && data.result.output_video) {
                    console.log('Final status:', data.result.output_video);
                    // Create a download link for the video file
                    window.location.href = data.output_audio;
                    //createDownloadLink(data.result.output_video);
                } else {
                    console.log('Task completed but no output video found.');
                }
            }
        })
        .catch(error => console.error('Error checking task status:', error));
}


if (window.location.pathname == '/doseedo1.html' || window.location.pathname == '/dashboard.html') {

    document.getElementById('description-input').addEventListener('change', function(event){

    });

document.getElementById('videoFile').addEventListener('change', function(event){
    const file = event.target.files[0];
    
    
    document.getElementById('showCuesBtn').style.visibility = 'visible'
    // document.querySelector('.video-container').style.width = '111%'

    

    
    if (file) {
        // Create a URL for the uploaded video file
    
        



        uploadVideo()
        // Hide or remove the upload field
        document.querySelector('.glow-container').style.display = 'none'; // Hides the "Upload a Video File" heading

    }
});












var videoFileElement = document.getElementById('videoFile');



document.getElementById('videoFile').addEventListener('mouseover', function(event){

    document.querySelector('.glow-container h4').style.visibility = 'visible'
});
document.getElementById('videoFile').addEventListener('mouseout', function(event){


    document.querySelector('.glow-container h4').style.visibility = 'hidden'
});

var videoFileElement = document.getElementById('videoFile');
var icon = document.querySelector('.glow-container i');

videoFileElement.addEventListener('dragover', function(event) {

    icon.classList.remove('shrinking-to-center');
    icon.classList.add('expanding-from-center');
    document.querySelector('.glow-container');
});

videoFileElement.addEventListener('mouseover', function(event) {

    icon.classList.remove('shrinking-to-center');
    icon.classList.add('expanding-from-center');
});

videoFileElement.addEventListener('dragleave', resetAnimation);
videoFileElement.addEventListener('drop', resetAnimation);
videoFileElement.addEventListener('mouseout', resetAnimation);


if(window.location.pathname == '/dashboard.html'){

const videoFileElement2 = document.getElementById('glow-container3');
const icon2 = document.querySelector('#glow-container3 i');

const videoFileElement3 = document.getElementById('glow-container4');
const icon3 = document.querySelector('#glow-container4 i');

const videoFileElement4 = document.getElementById('glow-container5');
const icon4 = document.querySelector('#glow-container5 i');




}




function resetAnimation() {
    icon.classList.remove('expanding-from-center');
    icon.classList.add('shrinking-to-center');
}

function onSignIn(googleUser) {
    var profile = googleUser.getBasicProfile();
    
    // For registration
    try{    
        registerUser(profile);

    }catch{
        loginUser(profile);



    // Or for login

}
}


}

let mode = 'video'
function switchto(){

    if(mode === 'text'){
        document.querySelector('.custom-file-input').style.visibility = 'block'

        
        document.getElementById('description-input').style.display = 'none'  
        document.querySelector('.glow-container i').style.visibility = 'visible'

        document.querySelector('#videoicon').style.display = 'none'
        document.querySelector('#keybicon').style.display = 'block'



        mode = 'video'
        console.log(mode)
        
    }else{
    document.querySelector('.glow-container i').style.visibility = 'hidden'
    document.querySelector('.custom-file-input').style.visibility = 'none'
    document.getElementById('description-input').style.display = 'block'

    document.querySelector('#keybicon').style.display = 'none'
    document.querySelector('#videoicon').style.display = 'block'
    mode = 'text'

    console.log(mode)

    
    }

}



let isVideoPlaying = false; // State variable to track play/pause status
let spacevid = document.getElementById('spaceaudio'); // Assuming this is the ID of your video element

let spacevid2 = document.querySelector('.spacevideo'); // Assuming this is the ID of your video element

let desertvid = document.getElementById('desertvid'); // Assuming this is the ID of your video element
  
//var homeLinksDiv = document.getElementById('download-links') // gets the first element
//var //button = homeLinksDiv.getElementsByTagName('button')[0] // gets the first button
//console.log(button)
const spacediv = document.getElementsByClassName('spacediv');
const spacediv2 = document.getElementsByClassName('spacediv2');



function playpreview() {
    const spacediv = document.getElementsByClassName('spacediv');

    let playButton = document.getElementById('play');
    let icon = document.getElementById('pauseplay'); // The icon element

    if (!isVideoPlaying) {
        // If the video is not playing, play it and switch icon to pause
        spacevid.play();
        icon.classList.remove('fa-play');
        icon.classList.add('fa-pause');
        isVideoPlaying = true;
        setTimeout(function(){
            icon.classList.add('fa-play');
            icon.classList.remove('fa-pause');
            isVideoPlaying = false
            


        }, ( (spacevid.duration - (spacevid.currentTime/0.7))*1000))
    } else {
        // If the video is playing, pause it and switch icon to play
        spacevid.pause();
        

        icon.classList.remove('fa-pause');
        icon.classList.add('fa-play');
        isVideoPlaying = false;
    }
}
function playpreview3() {
    const examplevid = document.getElementsByClassName('examplevid');

    let playButton = document.getElementById('play3');
    let icon = document.getElementById('pauseplay3'); // The icon element

    if (!isVideoPlaying) {
        // If the video is not playing, play it and switch icon to pause
        examplevid[currentSlide].play();
        icon.classList.remove('fa-play');
        icon.classList.add('fa-pause');
        isVideoPlaying = true;
        swiper.autoplay.stop();
        setTimeout(function(){
            icon.classList.add('fa-play');
            icon.classList.remove('fa-pause');
            isVideoPlaying = false
            swiper.autoplay.start();
            
            


        }, ( (examplevid[currentSlide].duration - (examplevid[currentSlide].currentTime))*1000))
    } else {
        // If the video is playing, pause it and switch icon to play
        examplevid[currentSlide].pause();
        

        icon.classList.remove('fa-pause');
        icon.classList.add('fa-play');
        swiper.autoplay.start();
        isVideoPlaying = false;
    }
}


let videx = 0; // Global index tracker
function fadeIn(element) {
    let op = 0.1;  // initial opacity
    element.style.opacity = op;
    let timer = setInterval(function () {
        if (op >= 1){
            clearInterval(timer);
        }
        element.style.opacity = op;
        element.style.filter = 'alpha(opacity=' + op * 100 + ")";
        op += op * 0.1;
    }, 10);
}
let videx2 = 0

let vidarray2 = ['desert.mov', 'beach.mov', 'monkeys.mov']
let promptarray = ['Desert Flute', 'Tropical Reggae', 'Kalimba and Tribal Drums']
let audioarray = ['2 (4).wav', '4 (3).wav', '0 (12).wav']
let fakeaudio = ''







function updateVideoSrc(videoElement, newSrc, animationClass) {
    videoElement.src = newSrc; // Update the src before starting the animation
    videoElement.classList.add(animationClass);
    videoElement.addEventListener('animationend', function() {
        videoElement.classList.remove(animationClass);
    }, { once: true });
}





let isRecording = false;
let mediaRecorder;
let audioChunks = [];
let stream; // To hold the media stream

if (window.location.pathname == '/doseedo1.html') {
    // Your logic here


    

    const recordToggle = document.getElementById("recordToggle");
    const recordIcon = document.getElementById("recordIcon");
    const audioPlayback = document.getElementById("audioPlayback");
    const melodyInput = document.getElementById("melodyInput");
    const uploadLabel2 = document.getElementById("uploadLabel2");
    
    recordToggle.addEventListener("click", async () => {
        if (!isRecording) {
            // Start Recording
            isRecording = true;
            recordIcon.className = "fa-regular fa-circle-stop";
            recordToggle.innerHTML = '<i id="recordIcon" class="fa-regular fa-circle-stop"></i> Stop';
    
            audioChunks = [];
            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
    
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };
    
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks);
                const audioUrl = URL.createObjectURL(audioBlob);
                audioPlayback.src = audioUrl;
    
                stream.getTracks().forEach(track => track.stop());
    
                melodyInput.style.display = 'none';
                audioPlayback.style.display = 'block';
                if (uploadLabel2) uploadLabel2.style.display = 'none';
            };
    
            mediaRecorder.start();
    
        } else {
            // Stop Recording
            isRecording = false;
            recordIcon.className = "fa-solid fa-microphone";
            recordToggle.innerHTML = '<i id="recordIcon" class="fa-solid fa-microphone"></i> Record';
    
            if (mediaRecorder?.state === "recording") {
                mediaRecorder.stop();
            }
        }
    });



}

function submitted(){
    document.getElementById('submitclick').value = 'Submitted.'

    setTimeout(() => {
        document.getElementById('submitclick').value = 'Submit'
        
    }, 2000);


}



if (window.location.pathname == '/home.html') {
    if (screen.width <= 700) {
    document.querySelector('.examplesdivmobile').style.display = 'block'
    document.querySelector('.examplesdiv').style.display = 'none'
    document.querySelector('.examplesdivmobile').style.width




    document.body.style.marginLeft = '150px';

    document.querySelector('#background').style.position = 'relative';
    document.querySelector('#background').style.right = '100px';
    document.querySelector('#background').style.width = '80%';

  

    }else{
        document.querySelector('.examplesdivmobile').style.display = 'none'
        document.querySelector('.examplesdiv').style.display = 'block'

    }






    


   //createDownloadLinks(['/download/8713ef2a-9679-4597-b5c2-756cea69584f/0.wav']); // Add your file paths here

}

let mousex = 0
let sleepmode = mousex
let clicked = false


function showdash() {
    const dashboard = document.getElementById('dashboard');
    const navlink = document.getElementsByClassName('nav-link');
  
    if (!clicked) {
      navlink[0].style.visibility = 'hidden';
      navlink[1].style.visibility = 'hidden';
      navlink[2].style.visibility = 'hidden';

      clicked = true;
    } else {
        navlink[0].style.visibility = 'visible';
        navlink[1].style.visibility = 'visible';
        navlink[2].style.visibility = 'visible';
  
      clicked = false;
    }
  }









document.addEventListener('DOMContentLoaded', (event) => {


    










    var videos = document.getElementsByTagName('video');
    
    // Loop through each video element and pause it
    for (var i = 0; i < videos.length; i++) {
        videos[i].pause();
    }
    
    let spacevid = document.getElementById('spacevid'); // Assuming this is the ID of your video element

    // Check the current time and set the video source based on day or night
    const currentTime = new Date();
    const hours = currentTime.getHours();
    const isDayTime = hours > 6 && hours < 18;

    



if (window.location.pathname == '/account.html') {
    document.getElementById('unsubscribeBtn').addEventListener('click', function(event) {
        event.preventDefault();
    
        // Confirm dialog
        var userConfirmed = confirm("Are you sure you want to unsubscribe?");
        
        // Proceed only if the user confirms
        if (userConfirmed) {
            var formData = new FormData();
            formData.append('email', using);
    
            fetch('http://doseedo.com/unsubscribe/', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => alert(data.message))
            .catch(error => console.error('Error:', error));
        }
    });
    

}


    if (window.location.pathname == '/home.html') {



    // Set the video source
    //spacevid.src = isDayTime ? 'examplesspace.mov' : 'examplesspace.mov';

    console.log(isDayTime)
    }

    // Ensure light mode is the default
    document.body.classList.remove('dark-mode');

    const themeSwitch = document.getElementById('theme-switch');

    // Set the switch state to match the default theme (light mode)
    //themeSwitch.checked = !isDayTime;
    //document.body.classList.toggle('dark-mode');

});









// Call this function to initially populate the projects list

