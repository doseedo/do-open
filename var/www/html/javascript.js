// JavaScript functions for showing/hiding forms and handling login
//Hey?

//TESTT

if (window.location.pathname === '/') {
    if (screen.width <= 700) {
        window.location.href = '/doseedo1.html';
        // console.log('test')
    }
}


let isFullDuration = false;
let isAuthenticated = false
let using = ''
let isPro = false
let mus = false
let currentVideoId = '' // Global variable to store the video ID
let videolabels = ''
let originalLengthInSeconds = 10
let currentZoomLevel = 1;
let selectedTracks = [];

let historyStack = [];
let redoStack = [];
let totalDuration = 10;
let audioContext = null
let zoomSlider = document.createElement("input");

    const reverbSlider = document.getElementById('reverb-slider');
    const fadeOutSlider = document.getElementById('fade-out-slider');
    const fadeInSlider = document.getElementById('fade-in-slider');




function saveHistory(action) {
    historyStack.push(JSON.parse(JSON.stringify(action)));
    redoStack = []; // Clear redo stack after new action
}



let splitScenes = false;
let useManualDuration   = false;

let globalSceneChanges = [];
let syncToSceneChanges = true;

let showMillisecondTicks = false;


let reverbIRBuffer = null;


async function loadImpulseResponse() {
    const ctx = initAudioContext();
    const response = await fetch('/impulses/1a_marble_hall.wav');
    const arrayBuffer = await response.arrayBuffer();
    reverbIRBuffer = await ctx.decodeAudioData(arrayBuffer);
    console.log('✅ Reverb IR loaded');
}





// 🔐 Early auth gate (must be very first lines)
const isAuth = localStorage.getItem('isAuth') === 'true';
const currentPath = window.location.pathname;


if (
  !isAuth &&
  currentPath !== '/login.html' &&
  currentPath !== '/forgotpassword.html'
) {
//   window.location.href = '/login.html';
  console.log("NOT AUTH!")
}




const loading = document.getElementById('loading')





// First, get a reference to the text input element
let currentSlide = 0;


if (window.location.pathname == '/doseedo1.html' ) {

const swiper = new Swiper('.swiper-container', {
    loop: true,
    autoplay: {
        delay: 5000,
        disableOnInteraction: false
    },
    navigation: {
        nextEl: '.swiper-button-next',
        prevEl: '.swiper-button-prev',
    },
});

swiper.on('slideChange', () => {
    currentSlide = swiper.realIndex;
});


}






// Listen for scroll events
function initDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('DoseedoDB', 2);
        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            if (!db.objectStoreNames.contains('videos')) {
                db.createObjectStore('videos');
            }
        };
        request.onsuccess = (event) => resolve(event.target.result);
        request.onerror = () => reject("❌ Failed to open DB");
    });
}

async function saveVideoBlobToIndexedDB(blob, key) {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('DoseedoDB', 2);

    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('videos')) {
        db.createObjectStore('videos');
        console.log("📂 Created 'videos' store in IndexedDB");
      }
    };

    request.onsuccess = (e) => {
      const db = e.target.result;
      const tx = db.transaction(['videos'], 'readwrite');
      const store = tx.objectStore('videos');

      const putReq = store.put(blob, key);
      putReq.onsuccess = () => {
        console.log("✅ Saved video blob with key:", key);
        resolve(true);
      };
      putReq.onerror = (e) => reject(e.target.error || e);
    };

    request.onerror = () => reject("❌ Failed to open IndexedDB");
  });
}





async function loadSession() {
    const projectName = localStorage.getItem('activeProject') || 'autosave';
    const stateStr = localStorage.getItem(`session-${projectName}`);
    if (!stateStr) return;

    const state = JSON.parse(stateStr);
    document.getElementById('session').textContent = projectName || 'Untitled Session';
    globalSceneChanges = state.sceneChanges || [];
    const containerWidth = downloadList.offsetWidth * currentZoomLevel;


    // renderSceneRangeBars(globalSceneChanges);
    // renderTimeline(originalLengthInSeconds, currentZoomLevel, containerWidth);
    const db = await initDB();
    const tx = db.transaction(['videos'], 'readonly');
    const store = tx.objectStore('videos');

    const video = document.getElementById('player');
    if (video && state.previewSrcKey) {
        const req = store.get(state.previewSrcKey);
        req.onsuccess = () => {
            const blob = req.result;
            if (blob) {
                const url = URL.createObjectURL(blob);
                video.src = url;
                video.muted = true;
                video.load();
                video.style.display = 'block';
                video.style.opacity = '1';
                zoomSlider.value = "1";
                zoomSlider.dispatchEvent(new Event('input'));

                const vidx = document.getElementById('vidx');
                vidx.style.display = 'block';

                
                video.onloadedmetadata = () => {
                    originalLengthInSeconds = video.duration || state.originalLengthInSeconds || 8;
                    totalDuration = originalLengthInSeconds;

                    // 🔍 Zoom out fully
                    currentZoomLevel = 1;


                    // videoPreviewsrc.muted()

                    document.querySelector('.glow-container').style.display = 'none'

                    const containerWidth = document.getElementById('download-links').offsetWidth * currentZoomLevel;
                    renderTimeline(originalLengthInSeconds, currentZoomLevel, containerWidth);
                    renderSceneRangeBars(globalSceneChanges);
                    resizeCanvas();
                    draw();


                
                };



            } else {
                originalLengthInSeconds = state.originalLengthInSeconds || 8;
                // totalDuration = originalLengthInSeconds; 
                // console.log(`🎬 Duration set from video metadata: ${totalDuration}s`);
            }
        };
        req.onerror = () => {
            console.warn("⚠️ Failed to load video blob");
            originalLengthInSeconds = state.originalLengthInSeconds || 8;
        };
    } else {
        originalLengthInSeconds = state.originalLengthInSeconds || 8;
    }

    for (const track of state.tracks) {
        window.trackAppendMode = true;
        await createDownloadLinks([track.fileUrl], [track.startTime], track.containerId, false);

        
        const container = document.getElementById(track.containerId);
        const lastLi = container?.querySelector('li:last-child');
        if (!lastLi || !lastLi._meta) continue;

        Object.assign(lastLi._meta, {
            startPositionInSeconds: track.startTime,
            trackDuration: track.duration,
            trimStart: track.trimStart,
            trimEnd: track.trimEnd
        });

        lastLi.dataset.uuid = track.trackId;

        const waveform = lastLi.querySelector('.waveform-container');
        const visual = track.visual || {};
        if (waveform && visual.width) waveform.style.width = visual.width;
        if (visual.startLineLeft) waveform.querySelector('.start-line')?.style.setProperty('left', visual.startLineLeft);
        if (visual.endLineLeft) waveform.querySelector('.end-line')?.style.setProperty('left', visual.endLineLeft);
        if (visual.leftMaskWidth) waveform.querySelector('.waveform-mask-left')?.style.setProperty('width', visual.leftMaskWidth);
        if (visual.rightMaskWidth) {
            const rightMask = waveform.querySelector('.waveform-mask-right');
            rightMask.style.width = visual.rightMaskWidth;
            rightMask.style.left = `calc(100% - ${visual.rightMaskWidth})`;
        }

        if (typeof updateTrackPosition === 'function') {
            updateTrackPosition(lastLi);
        }
    }

    // const containerWidth = document.getElementById('download-links').offsetWidth * currentZoomLevel;
    const pixelsPerSecond = getPixelsPerSecond();


    if (selectedTracks.length === 1) {
        updateSliderUIFromSelectedTrack();
    }


    // if (Array.isArray(state.cues)) {
    //     document.querySelectorAll('.scene-arrow').forEach(el => el.remove());
    // state.cues.forEach(cue => {
    //     const arrow = renderCueArrow(parseFloat(cue.time), cue.label, cue.type, pixelsPerSecond);
    //     document.getElementById('timeline-bar')?.appendChild(arrow);
    //     allCues.push(arrow);
    // });
    // }

    

    if (state.historyStack) historyStack = state.historyStack;
    if (state.redoStack) redoStack = state.redoStack;

    


    window.trackAppendMode = false;
    console.log(`✅ Loaded session ${projectName}`);
}























window.onload = function() {

    

    
    const isAuth = localStorage.getItem('isAuth') === 'true';
    const username = localStorage.getItem('username');
    const substatus = localStorage.getItem('ispro');
    const currentPath = window.location.pathname;

    const userPic = localStorage.getItem('userpic');
        if (userPic) {
            const userPicElement = document.getElementById('userpic');
            if (userPicElement) {
                userPicElement.src = userPic;
                userPicElement.style.display = 'block';
            }
}


    using = username;

    // 🔒 Redirect if not authenticated
    if (
        !isAuth &&
        currentPath !== '/login.html' &&
        currentPath !== '/forgotpassword.html'
        ) {
        // window.location.href = '/login.html';
        console.log("NOTAUTH!")
        }

    // ✅ If already logged in and on login page, send to dashboard
    if (isAuth && currentPath === '/login.html') {
        window.location.href = '/doseedo1.html';
        return;
    }

    // if (!isAuth) {
    //     window.location.href = '/login.html';

    // }


    // 👤 UI setup for authenticated user
    if (isAuth && username) {
        isAuthenticated = true;

        const loginBtn = document.getElementById('loginbutton');
        if (loginBtn) loginBtn.style.display = 'none';

        if (substatus === 'Pro+') {
            isPro = true;
        }

        try {
            document.getElementById('user-username').textContent = username;
            document.getElementById('user-subscription-status').textContent = substatus;
            document.getElementById('user-info').style.display = 'flex';
            // document.getElementById('register-form').style.display = 'none';
            // document.getElementById('login-form').style.display = 'none';
            // document.getElementById('signupbutton').style.display = 'none';
        } catch (error) {
            console.warn("UI update error:", error);
        }
    }

      initAudioContext();





    // 🧾 Show plan status on /plans.html
    // if (currentPath === '/plans.html') {
    //     if (!isAuth) {
    //         document.querySelector('.notsubbed')?.style.display = 'none';
    //         document.querySelector('.notsignedin')?.style.display = 'block';
    //     } else if (substatus === 'Pro+') {
    //         document.querySelector('.notsubbed')?.style.display = 'none';
    //         document.querySelector('.subbed')?.style.display = 'block';
    //         document.querySelector('.notsignedin')?.style.display = 'none';
    //     }
    // }
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
    
    const notsubbedEls = document.getElementsByClassName('notsubbed');
for (let el of notsubbedEls) {
    el.style.display = 'none';
}

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
textInput.addEventListener('input', handleInput);


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
    // document.getElementById('user-info').style.display = 'none';
    document.body.style.bottom = '100px'
    // document.getElementById('background').style.opacity = '0.1'
}



function showSignIn() {
    document.getElementById('register-form').style.display = 'none';
    document.querySelector('.formdiv').style.display = 'block';
    //document.querySelector('.formdiv').style.display = 'block';
    document.getElementById('login-form').style.display = 'block';
    // document.getElementById('user-info').style.display = 'none';

    // document.getElementById('background').style.opacity = '0.1'


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

    localStorage.setItem('isAuth', 'true');
    localStorage.setItem('username', username);
    localStorage.setItem('ispro', 'Free'); // or 'Pro+' if that's your logic
    localStorage.setItem('userpic', user.getPicture() || '');


    // document.getElementById('user-username').textContent = user.username;
    // document.getElementById('user-info').style.display = 'block';
    document.getElementById('register-form').style.display = 'none';
    document.getElementById('login-form').style.display = 'none';
    window.location.href = '/doseedo1.html'
    // document.querySelector('.guest').style.display = 'none';

}

function registerUser() {
    const username = document.getElementById('register-username').value;
    const email = document.getElementById('register-email')
    const password = document.getElementById('register-password').value;
    alert(`User registered successfully!\nUsername: ${username}\nEmail: ${email}`);
}
function nextstep() {
    // const username = document.getElementById('register-username').style.display = 'inline';
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
        const isRegen = typeof window._regenInsertIndex === 'number' && selectedTracks.length === 1;
        const selectedIndex = window._regenInsertIndex;
        const sceneStarts = globalSceneChanges;
    
        // Cleanup regen flags
        delete window._regenInsertIndex;
        delete window._regenInsertTop;
        delete window._regenContainerId;
        delete window._regenDuration;
    
        const formData = new FormData();
    
        if (isRegen) {
            const selectedTrack = selectedTracks[0];
            const sceneIndex = parseInt(selectedTrack.dataset.sceneIndex, 10);
    
            const sceneStart = sceneStarts[sceneIndex];
            const sceneEnd = sceneIndex < sceneStarts.length - 1
                ? sceneStarts[sceneIndex + 1]
                : originalLengthInSeconds;
            const sceneDuration = sceneEnd - sceneStart;
    
            const label = selectedTrack.dataset.originalLabel?.trim() || 'Regenerated';
    
            formData.append('labels_raw', JSON.stringify([label]));
            formData.append('scene_durations', JSON.stringify([sceneDuration]));
            formData.append('video_output', label);
            const directAutomation = document.getElementById("useDirectAutomation").checked;
            formData.append("use_direct_automation", directAutomation.toString());

        } else {
            const descriptionInput = document.getElementById('description-input');
            const labels = descriptionInput.value.split(', ');
            const sceneDurations = [];
    
            for (let i = 0; i < sceneStarts.length; i++) {
                const start = sceneStarts[i];
                const end = i < sceneStarts.length - 1 ? sceneStarts[i + 1] : originalLengthInSeconds;
                sceneDurations.push(end - start);
            }
    
            formData.append('labels_raw', JSON.stringify(labels));
            formData.append('scene_durations', JSON.stringify(sceneDurations));
            formData.append('video_output', descriptionInput.value);
            const directAutomation = document.getElementById("useDirectAutomation").checked;
            formData.append("use_direct_automation", directAutomation.toString());
        }
    
        formData.append('request_type', 'normal');
        formData.append('split_scenes', splitScenes);
        const promptValue = document.getElementById('prompt-select')?.value || '';
        if (promptValue) {
            formData.append('prompt', promptValue);
        }
    
        updateStatus('Generating SFX…');
        loading.style.display = 'block';
        document.querySelector('#gensfx').style.pointerEvents = 'none';
        document.querySelector('#gensfx').style.opacity = '20%';
        document.querySelector('#genmus').style.pointerEvents = 'none';
        document.querySelector('#genmus').style.opacity = '20%';
    
        fetch('/generate', {
            method: 'POST',
            body: formData
        })
        .then(r => r.json())
        .then(data => {
            if (data.task_id) {
                const callback = (urls) => {
                    const startTimes = isRegen ? [sceneStarts[selectedIndex]] : sceneStarts;
                    createDownloadLinks(urls, startTimes, 'download-links2');
                };
                pollTaskStatus(data.task_id, callback);
            }
        })
        .catch(err => {
            console.error("❌ Error generating audio:", err);
            updateStatus("Error generating audio.");
            loading.style.display = 'none';
            document.querySelector('#gensfx').style.pointerEvents = 'auto';
            document.querySelector('#gensfx').style.opacity = '100%';
            document.querySelector('#genmus').style.pointerEvents = 'auto';
            document.querySelector('#genmus').style.opacity = '100%';
        });
    }
    
    



// let splitScenes = /* document.getElementById('splitScenesCheckbox').checked*/ 

 

function getCleanAutomationData() {
    const sortedPoints = [...automationPoints].sort((a, b) => a.x - b.x);
    return sortedPoints.map(point => ({
        // Convert X to absolute seconds in video timeline
        time: (point.x / canvas.width) * totalDuration,
        // Convert Y to 0-1 volume (0=bottom, 1=top)
        volume: 1 - (point.y / canvas.height)
    }));
}



function generateMusic() {
    const isRegen = typeof window._regenInsertIndex === 'number' && selectedTracks.length === 1;
    const selectedIndex = window._regenInsertIndex;
    const sceneStarts = globalSceneChanges;

    delete window._regenInsertIndex;
    delete window._regenInsertTop;
    delete window._regenContainerId;
    delete window._regenDuration;

    const params = new URLSearchParams();

    let rawSceneData = null;

    if (isRegen) {
        const selectedTrack = selectedTracks[0];
        const sceneIndex = parseInt(selectedTrack.dataset.sceneIndex, 10);

        if (sceneIndex >= sceneStarts.length) {
            console.error('Scene index out of bounds:', sceneIndex);
            return;
        }

        const sceneStart = sceneStarts[sceneIndex];
        const sceneEnd = sceneIndex < sceneStarts.length - 1 
            ? sceneStarts[sceneIndex + 1] 
            : originalLengthInSeconds;
        const sceneDuration = sceneEnd - sceneStart;

        const automation = {
            points: getCleanAutomationData(),
            resolution: { width: canvas.width, height: canvas.height }
        };

        const allSceneData = JSON.parse(localStorage.getItem("sceneData") || "[]");
        const sceneObj = allSceneData[sceneIndex] || {};
        rawSceneData = JSON.stringify([sceneObj]);

        params.set('automation_data', JSON.stringify(automation));
        params.set('scene_durations', JSON.stringify([sceneDuration]));
        params.set('request_type', 'melody');
        params.set('split_scenes', 'false');
        params.set('scene_index', sceneIndex);
        params.set('all_scene_changes', JSON.stringify(sceneStarts));
    } else {
        const sceneData = localStorage.getItem("sceneData");
        const sceneDurations = localStorage.getItem("sceneDurations");

        if (!sceneData || !sceneDurations) {
            console.error("Missing sceneData or sceneDurations in localStorage");
            return;
        }

        try {
            JSON.parse(sceneData);
            rawSceneData = sceneData;
        } catch (e) {
            console.error("Invalid sceneData:", e);
            return;
        }

        const automation = getAutomationData();

        params.set('automation_data', JSON.stringify(automation));
        params.set('scene_durations', sceneDurations);
        params.set('global_changes', globalSceneChanges);
        params.set('request_type', 'melody');
        params.set('split_scenes', splitScenes);
    }

    // UI state
    window.trackAppendMode = true;
    updateStatus('Generating music…');
    document.querySelector('#genmus').style.pointerEvents = 'none';
    document.querySelector('#genmus').style.opacity = '20%';
    loading.style.display = 'block';

    const fd = new FormData();
    params.forEach((v, k) => fd.append(k, v));

    if (rawSceneData) {
        fd.append('scene_data_raw', rawSceneData);
        console.log("✅ scene_data_raw contents:", rawSceneData);
    } else {
        console.error("❌ Missing scene_data_raw");
    }

    const useDirect = document.getElementById('useDirectAutomation');
    fd.append('use_direct_automation', useDirect?.checked ? 'true' : 'false');

    if (key) fd.append('prompt', keySelect.value);

    const melodyInput = document.getElementById('melodyInput');
    if (melodyInput?.files.length > 0) {
        fd.append('melody', melodyInput.files[0]);
    }

    // Debug
    console.log("🎯 FINAL REQUEST PAYLOAD:");
    for (let [k, v] of fd.entries()) console.log(k, v);

    fetch('https://doseedo.com/generate', {
        method: 'POST',
        body: fd
    })
    .then(response => response.json())
    .then(data => {
        if (data.task_id) {
            const callback = (urls) => {
                const startTimes = isRegen
                    ? [sceneStarts[selectedIndex]]
                    : sceneStarts;
                createDownloadLinks(urls, startTimes, 'download-links', isRegen);
            };
            pollTaskStatus(data.task_id, callback);
        }
    })
    .catch(err => {
        console.error("❌ Error during generation:", err);
        updateStatus("Error during generation");
        loading.style.display = 'none';
    });
}



function generateRisers() {
    const isRegen = typeof window._regenInsertIndex === 'number' && selectedTracks.length === 1;
    const selectedIndex = window._regenInsertIndex;
    const sceneStarts = globalSceneChanges;

    // Clear regen flags
    delete window._regenInsertIndex;
    delete window._regenInsertTop;
    delete window._regenContainerId;
    delete window._regenDuration;

    const descIn = document.getElementById('description-input').value;
    const params = new URLSearchParams();

    if (isRegen) {
        const selectedTrack = selectedTracks[0];
        const sceneIndex = parseInt(selectedTrack.dataset.sceneIndex, 10);

        if (sceneIndex >= sceneStarts.length) {
            console.error('Scene index out of bounds:', sceneIndex);
            return;
        }

        const sceneStart = sceneStarts[sceneIndex];
        const sceneEnd = sceneIndex < sceneStarts.length - 1
            ? sceneStarts[sceneIndex + 1]
            : originalLengthInSeconds;
        const sceneDuration = sceneEnd - sceneStart;

        const label = selectedTrack.dataset.originalLabel?.trim() || 'Regenerated Riser';

        const automation = {
            points: getCleanAutomationData(),
            resolution: { width: canvas.width, height: canvas.height }
        };

        params.set('automation_data', JSON.stringify(automation));
        params.set('labels_raw', JSON.stringify([label]));
        params.set('video_output', label);
        params.set('request_type', 'riser');
        params.set('split_scenes', 'false');
        params.set('scene_durations', JSON.stringify([sceneDuration]));
        params.set('scene_index', sceneIndex);
        params.set('all_scene_changes', JSON.stringify(globalSceneChanges));
    } else {
        const labels = descIn.split(', ');
        const sceneDurations = [];

        for (let i = 0; i < sceneStarts.length; i++) {
            const start = sceneStarts[i];
            const end = i < sceneStarts.length - 1 ? sceneStarts[i + 1] : originalLengthInSeconds;
            sceneDurations.push(end - start);
        }

        const automation = getAutomationData();
        params.set('automation_data', JSON.stringify(automation));
        params.set('labels_raw', JSON.stringify(labels));
        params.set('video_output', descIn);
        params.set('request_type', 'riser');
        params.set('split_scenes', splitScenes);
        params.set('scene_durations', JSON.stringify(sceneDurations));
    }

    // UI
    window.trackAppendMode = true;
    updateStatus('Generating risers…');
    loading.style.display = 'block';

    const fd = new FormData();
    params.forEach((v, k) => fd.append(k, v));

    if (key) fd.append('prompt', keySelect.value);

    const melodyInput = document.getElementById('melodyInput');
    if (melodyInput && melodyInput.files.length > 0) {
        fd.append('melody', melodyInput.files[0]);
    }

    console.log("🎯 RISER REQUEST PAYLOAD:");
    for (let [k, v] of fd.entries()) console.log(k, v);

    fetch('https://doseedo.com/generate', {
        method: 'POST',
        body: fd
    })
    .then(response => response.json())
    .then(data => {
        if (data.task_id) {
            const callback = (urls) => {
                let startTimes;
                if (isRegen) {
                    startTimes = [Math.max(0, sceneStarts[selectedIndex] - 2)];
                } else {
                    startTimes = sceneStarts
                        .slice(1)                        // Exclude first scene
                        .map(time => Math.max(0, time - 2));  // Offset by 2 sec
                }

                createDownloadLinks(urls, startTimes, 'download-links2', isRegen, { forceFullLength: true });

            };
            pollTaskStatus(data.task_id, callback);
        }
    })
    .catch(err => {
        console.error("❌ Error during riser generation:", err);
        updateStatus("Error during riser generation");
        loading.style.display = 'none';
    });
}





function regen() {
    if (selectedTracks.length !== 1) {
        alert("Please select exactly one track to regenerate.");
        return;
    }

    const track = selectedTracks[0];
    const parentList = track.closest('[id^="download-links"]');
    const sceneIndex = parseInt(track.dataset.sceneIndex, 10); 

    window._regenInsertIndex = sceneIndex;
    window._regenInsertTop = parseInt(track.dataset.originalTop, 10);
    window._regenContainerId = parentList.id;
    window._regenDuration = track._meta?.trackDuration ?? 4.0;
    window._regenReplaceIndexInDOM = Array.from(parentList.children).indexOf(track);
    window._regenReplaceElement = track;

    // Visual indicator
    const waveform = track.querySelector('.waveform-container');
    waveform.style.opacity = '0.3';
    const waveCanvas = waveform.querySelector('canvas');
    if (waveCanvas) waveCanvas.style.display = 'none';

    const loadingGif = document.createElement('img');
    loadingGif.src = 'loadinger.gif';
    loadingGif.className = 'regen-loading-indicator';
    loadingGif.style.position = 'absolute';
    loadingGif.style.top = '0';
    loadingGif.style.left = '0';
    loadingGif.style.width = '100%';
    loadingGif.style.height = '100%';
    loadingGif.style.objectFit = 'cover';
    loadingGif.style.zIndex = '2';
    waveform.appendChild(loadingGif);

    // 🔁 Decide what to regenerate

    const listId = parentList.id;

    if (listId === 'download-links') {
        generateMusic();
    } else if (listId === 'download-links2') {
        generateAudio();
    } else {
        alert("Unsupported track type for regeneration.");
    }
    }



  
  // You can leave your existing pollTaskStatus() in place for the one-shot path (no changes needed).
  
  

  document.querySelectorAll('.bus-mute').forEach(btn => {
    btn.addEventListener('click', (e) => {
        e.stopPropagation();  // ← prevent parent click

        const containerId = btn.dataset.bus;
        const container = document.getElementById(containerId);
        const isMuting = !btn.classList.contains('mutedbtn');

        btn.classList.toggle('mutedbtn', isMuting);

        container.querySelectorAll('li').forEach(track => {
            if (container.querySelector('.soloed')) return;

            track.classList.toggle('muted', isMuting);
            track._wavesurfer?.setVolume(isMuting ? 0 : 1);

            const waveform = track.querySelector('.waveform-container');
            waveform.style.background = isMuting ? '#666' : waveform.dataset.originalColor;
        });
    });
});

document.querySelectorAll('.bus-solo').forEach(btn => {
    btn.addEventListener('click', (e) => {
        e.stopPropagation();  // ← prevent parent click

        const containerId = btn.dataset.bus;
        const container = document.getElementById(containerId);
        const isSoloing = !btn.classList.contains('soloed');

        // Reset all solo buttons and volumes
        document.querySelectorAll('.bus-solo').forEach(b => {
            b.classList.remove('soloed');
            b.style.background = '';
        });

        document.querySelectorAll('[id^="download-links"]').forEach(dl => {
            dl.querySelectorAll('li').forEach(track => {
                track.classList.remove('muted');
                track._wavesurfer?.setVolume(1);
                track.querySelector('.waveform-container').style.opacity = 1;
            });
        });

        if (isSoloing) {
            btn.classList.add('soloed');
            btn.style.background = 'rgb(125 155 221 / 94%)';

            container.querySelectorAll('li').forEach(track => {
                track.classList.remove('mutedbtn');
                track._wavesurfer?.setVolume(1);
                track.querySelector('.waveform-container').style.opacity = 1;
            });

            // Mute all others
            document.querySelectorAll('[id^="download-links"]').forEach(dl => {
                if (dl.id === containerId) return;
                dl.querySelectorAll('li').forEach(track => {
                    track.classList.add('mutedbtn');
                    track._wavesurfer?.setVolume(0);
                    track.querySelector('.waveform-container').style.opacity = 0.5;
                });
            });
        }
    });
});






// let audioContext;

function initAudioContext() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        audioContext.onstatechange = () => {
            audioContextState = audioContext.state;
            console.log('AudioContext state:', audioContext.state);
        };
    }
    return audioContext;
}


function monitorAudioContext() {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    audioContext.onstatechange = () => {
        console.log('AudioContext state:', audioContext.state);
    };
    return audioContext;
}

// Initialize on first user interaction
// document.addEventListener('click', () => {
//     if (!audioContext) {
//         audioContext = monitorAudioContext();
//         // Resume context on first user click
//         audioContext.resume();
//     }
// }, { once: true });


// let reverbIRBuffer = null;

fetch('/impulses/1a_marble_hall.wav') // ✅ make sure this file exists
    .then(r => r.arrayBuffer())
    .then(buf => initAudioContext().decodeAudioData(buf))
    .then(decoded => {
        reverbIRBuffer = decoded;
        console.log("🎛️ Reverb IR loaded");
    });



const busNodes = {}; // Store shared GainNodes per container


function updateSliderUIFromSelectedTrack() {
    const track = getSelectedTrack();
    if (!track || !track._meta) return;

    const meta = track._meta;
    
    // Initialize default values if missing
    if (typeof meta.reverbAmount !== 'number') {
        meta.reverbAmount = 0.2;
    }
    if (typeof meta.fadeOutDuration !== 'number') {
        meta.fadeOutDuration = 2.0;
    }

    reverbSlider.value = meta.reverbAmount;
    fadeInSlider.value = meta.fadeInDuration;
    fadeOutSlider.value = meta.fadeOutDuration;
}



function setupBusGainControls() {
  const busMap = {
    'music-gain': 'download-links',
    'sfx-gain': 'download-links2',
    'vo-gain': 'download-links3'
  };

  Object.entries(busMap).forEach(([sliderId, containerId]) => {
    const slider = document.getElementById(sliderId);
    const container = document.getElementById(containerId);
    if (!slider || !container) return;

    slider.addEventListener('input', () => {
  const gain = parseFloat(slider.value);

  container.querySelectorAll('li').forEach(track => {
    const ws = track._wavesurfer;
    const fx = track._fx;

    // Adjust dry mix (WaveSurfer main gain)
    if (ws?.backend?.gainNode && !track.classList.contains('muted')) {
      ws.setVolume(gain);
    }

    // Adjust reverb bus if exists
    if (fx?.outputGain) {
      fx.outputGain.gain.value = gain;
    }
  });
});

  });
}





    function getSelectedTrack() {
    return selectedTracks.length === 1 ? selectedTracks[0] : null;
    }

  

  document.addEventListener('DOMContentLoaded', () => {
    setupBusGainControls();
    loadImpulseResponse();




// Enhanced slider event handlers
// Modify the reverb slider event listener to:

// Modified reverb slider handler
reverbSlider.addEventListener('input', () => {
    const track = getSelectedTrack();
    if (!track?._fx) return;

    // 1. Get valid audio context
    const ctx = initAudioContext();
    
    // 2. Validate and parse value
    let amt = parseFloat(reverbSlider.value);
    
    // Validate and constrain between 0-1
    if (isNaN(amt)) {
        console.warn('Invalid reverb value, resetting to 0.5');
        amt = 0.2;
        reverbSlider.value = amt;
    }
    amt = Math.min(1, Math.max(0, amt));
    
    // 3. Update metadata
    track._meta.reverbAmount = amt;

    try {
        // 4. Update audio nodes using proper context time
        const now = ctx.currentTime;
        track._fx.dryGain.gain.setValueAtTime(1 - amt, now);
        track._fx.wetGain.gain.setValueAtTime(amt, now);
    } catch (error) {
        console.error('Reverb error:', error);
        // Reset to default values
        track._fx.dryGain.gain.value = 0.8;
        track._fx.wetGain.gain.value = 0.2;
    }
});

['change', 'mouseup', 'mouseleave', 'touchend'].forEach(eventName => {
    reverbSlider.addEventListener(eventName, () => {
        const track = getSelectedTrack();
        if (!track?._meta) return;

        let amt = parseFloat(reverbSlider.value);
        if (!isNaN(amt)) {
            track._meta.reverbAmount = Math.min(1, Math.max(0, amt));
        }
    });
});

// Add this with other slider declarations
const handleReverbChange = debounce(() => {
    const track = getSelectedTrack();
    if (!track?._fx) return;

    const ctx = initAudioContext();
    const amt = Math.min(1, Math.max(0, parseFloat(reverbSlider.value)));
    
    // Cancel any automation
    track._fx.dryGain.gain.cancelScheduledValues(ctx.currentTime);
    track._fx.wetGain.gain.cancelScheduledValues(ctx.currentTime);
    
    // Set immediate values
    track._fx.dryGain.gain.value = 1 - amt;
    track._fx.wetGain.gain.value = amt;
    
    track._meta.reverbAmount = amt;
}, 50);

reverbSlider.addEventListener('input', handleReverbChange);


// Add this in setupGlobalControls() where controls are initialized
const fadeInSlider = document.getElementById('fade-in-slider'); // Ensure correct ID

const fadeOutSlider = document.getElementById('fade-out-slider'); // Ensure correct ID
const fadeToEndCheckbox = document.getElementById('fade-to-end-line');



fadeToEndCheckbox.addEventListener('change', () => {
    const fadeValue = parseFloat(fadeOutSlider.value);
    const fadeToEnd = fadeToEndCheckbox.checked;

    document.querySelectorAll('.waveform-container.selected').forEach(selectedWaveform => {
        const track = selectedWaveform.closest('li');
        if (!track || !track._meta) return;

        const meta = track._meta;
        meta.fadeOutDuration = fadeValue;
        meta.fadeToEndLine = fadeToEnd;

        if (!track._fx || !track._fx.masterGain) {
            console.warn("⏸ No audio playing — fade setting saved and will apply on next playback.");
            return;
        }

        const ctx = initAudioContext();
        const now = ctx.currentTime;
        const currentTime = videoPreviewsrc.currentTime;

        const trimOffsetStart = meta.startPositionInSeconds + meta.trimStart;
        const offset = Math.max(0, currentTime - trimOffsetStart);

        const fadeTargetEnd = fadeToEnd
            ? meta.trimEnd
            : Math.min(meta.trackDuration + fadeValue, track._decodedBuffer.duration);

        const fadeStartOffset = fadeTargetEnd - fadeValue;
        const duration = Math.min(
            fadeTargetEnd - meta.trimStart - offset,
            track._decodedBuffer.duration - meta.trimStart
        );

        const fadeStartTime = now + (fadeStartOffset - offset);

        const gain = track._fx.masterGain.gain;
        gain.cancelScheduledValues(now);
        gain.setValueAtTime(1.0, now);

        if (fadeValue > 0.05 && fadeStartTime > now) {
            gain.exponentialRampToValueAtTime(0.001, fadeStartTime + fadeValue);
        } else {
            gain.exponentialRampToValueAtTime(0.001, now + 0.1);
        }

        console.log(`🎚 Fade-out updated to ${fadeValue}s (${fadeToEnd ? '→ trimEnd' : '→ after track end'})`);
    });
});




fadeOutSlider.addEventListener('input', () => {
    const fadeValue = parseFloat(fadeOutSlider.value);
    const fadeToEnd = fadeToEndCheckbox.checked;

    document.querySelectorAll('.waveform-container.selected').forEach(selectedWaveform => {
        const track = selectedWaveform.closest('li');
        if (track?._meta) {
            track._meta.fadeOutDuration = fadeValue;
            track._meta.fadeToEndLine = fadeToEnd; // ✅ Store checkbox value
        }
    });
});

// 🟢 On change: update playback automation
fadeOutSlider.addEventListener('change', () => {
    const fadeValue = parseFloat(fadeOutSlider.value);
    const fadeToEnd = fadeToEndCheckbox.checked;

    document.querySelectorAll('.waveform-container.selected').forEach(selectedWaveform => {
        const track = selectedWaveform.closest('li');
        if (!track?._meta || !track._fx || !track._fx.masterGain) return;

        const meta = track._meta;
        meta.fadeOutDuration = fadeValue;
        meta.fadeToEndLine = fadeToEnd;

        const ctx = initAudioContext();
        const now = ctx.currentTime;

        const trimOffsetStart = meta.startPositionInSeconds + meta.trimStart;
        const currentTime = videoPreviewsrc.currentTime;
        const offset = Math.max(0, currentTime - trimOffsetStart);

        const fadeTargetEnd = fadeToEnd ? meta.trimEnd : meta.trackDuration;
        const duration = Math.min(
            fadeTargetEnd - meta.trimStart - offset,
            track._decodedBuffer.duration - meta.trimStart
        );

        const fadeStartTime = now + (duration - fadeValue);

        track._fx.masterGain.gain.cancelScheduledValues(now);
        track._fx.masterGain.gain.setValueAtTime(1.0, now);

        if (fadeValue > 0.05 && fadeStartTime > now) {
            track._fx.masterGain.gain.exponentialRampToValueAtTime(
                0.001, fadeStartTime + fadeValue
            );
        } else {
            track._fx.masterGain.gain.exponentialRampToValueAtTime(0.001, now + 0.1);
        }

        console.log(`🎚 Fade-out updated to ${fadeValue}s (${fadeToEnd ? '→ trimEnd' : '→ full duration'})`);
    });
});

// 🔄 Same pattern for fade-in
fadeInSlider.addEventListener('input', () => {
    const fadeValue = parseFloat(fadeInSlider.value);
    document.querySelectorAll('.waveform-container.selected').forEach(selectedWaveform => {
        const track = selectedWaveform.closest('li');
        if (track?._meta) {
            track._meta.fadeInDuration = fadeValue;
        }
    });
});

fadeInSlider.addEventListener('change', () => {
    const fadeValue = parseFloat(fadeInSlider.value);
    document.querySelectorAll('.waveform-container.selected').forEach(selectedWaveform => {
        const track = selectedWaveform.closest('li');
        if (!track?._meta || !track._fx || !track._fx.masterGain) return;

        const meta = track._meta;
        meta.fadeInDuration = fadeValue;

        const ctx = initAudioContext();
        const now = ctx.currentTime;

        track._fx.masterGain.gain.cancelScheduledValues(now);

        if (fadeValue > 0.05) {
            track._fx.masterGain.gain.setValueAtTime(0.001, now);
            track._fx.masterGain.gain.exponentialRampToValueAtTime(1.0, now + fadeValue);
        } else {
            track._fx.masterGain.gain.setValueAtTime(1.0, now);
        }

        console.log(`🎚 Fade-in updated to ${fadeValue}s`);
    });
});





  });

  






// console.log('settings clicked')

setTimeout(() => {
    document.getElementById('settings').click()
}, 100);


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


const WAVEFORM_ACTIVE_COLOR = 'rgb(15 15 15)';
const WAVEFORM_SELECTED_OPACITY = '70%';
const WAVEFORM_UNSELECTED_OPACITY = '100%';
const WAVEFORM_SELECTED_WIDTH = '5px';
const WAVEFORM_UNSELECTED_WIDTH = '2px';

document.addEventListener('mousedown', (e) => {
    if (
        e.target.closest('.start-line') ||
        e.target.closest('.end-line') ||
        e.target.closest('.start-handle') ||
        e.target.classList.contains('interact-resize-handle') ||
        e.target.classList.contains('resizer') ||
        e.target.closest('.resizer')
    ) return;

    const clickedContainer = e.target.closest('.waveform-container');
    if (!clickedContainer) return;

    const listItem = clickedContainer.closest('li');
    const downloadList = listItem.closest('[id^="download-links"]');
    const isCollapsed = !downloadList.classList.contains('expanded');
    const isShift = e.shiftKey;
    e.preventDefault();

    const allContainers = document.querySelectorAll('.waveform-container');
    const containerWidth = downloadList.offsetWidth * currentZoomLevel;
    const pixelsPerSecond = getPixelsPerSecond();


    const meta = listItem._meta;
    const leftMask = clickedContainer.querySelector('.waveform-mask-left');
    const rightMask = clickedContainer.querySelector('.waveform-mask-right');

    const cropLeftPx = meta.trimStart * pixelsPerSecond;
    const cropRightPx = (meta.trackDuration - meta.trimEnd) * pixelsPerSecond;
    const trackWidthPx = meta.trackDuration * pixelsPerSecond;

    if (!isShift) {
        const wasSelected = clickedContainer.classList.contains('selected');

        allContainers.forEach(container => {
            container.classList.remove('selected');
            const li = container.closest('li');
            if (li) li.style.zIndex = '0';
        });

        // Always select clicked track
        clickedContainer.classList.add('selected');
        document.getElementById('fx-panel').style.display = 'block'
        listItem.style.zIndex = 9999;

        if (isCollapsed) {
            if (leftMask) {
                leftMask.style.display = 'block';
                leftMask.style.left = '0px';
                leftMask.style.width = `${cropLeftPx}px`;
            }

            if (rightMask) {
                rightMask.style.display = 'block';
                rightMask.style.left = `${trackWidthPx - cropRightPx}px`;
                rightMask.style.width = `${cropRightPx}px`;
            
            }
        }
    } else {
        const nowSelected = clickedContainer.classList.toggle('selected');
        listItem.style.zIndex = nowSelected ? 9999 : '0';
                document.getElementById('fx-panel').style.display = 'none'

        if (isCollapsed) {
            if (nowSelected) {
                if (leftMask) {
                    leftMask.style.display = 'block';
                    leftMask.style.left = '0px';
                    leftMask.style.width = `${cropLeftPx}px`;
                    
                }

                if (rightMask) {
                    rightMask.style.display = 'block';
                    rightMask.style.left = `${meta.trimEnd * pixelsPerSecond}px`;
                    rightMask.style.width = `${cropRightPx}px`
                }
            } else {
                adjustMasksOnCollapse(downloadList);
            }
        }
    }

    selectedTracks = Array.from(document.querySelectorAll('.waveform-container.selected'))
        .map(c => c.closest('li'));
    updateSliderUIFromSelectedTrack();

    });
    updateRegenButtonOpacity();















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
    loading.style.display = 'block';

    const videoInput = document.getElementById('videoFile');
    const videoElement = document.getElementById('player'); // make sure this is your video element
    const loaderOverlay = document.getElementById('videoLoader'); // this is the <img src="loading.gif"> you position over video

    if (videoInput.files.length === 0) {
        alert('Please select a video file.');
        return;
    }

    // Dim video and show loading overlay
    if (videoElement) videoElement.style.opacity = '0.5';
    if (loaderOverlay) loaderOverlay.style.display = 'block';

    // Disable buttons
    document.querySelector('#genmus').style.pointerEvents = 'none';
    document.querySelector('#genmus').style.opacity = '20%';
    document.querySelector('#gensfx').style.pointerEvents = 'none';
    document.querySelector('#gensfx').style.opacity = '20%';

    const videoFile = videoInput.files[0];

        const formData = new FormData();
    formData.append('file', videoFile);

    const blob = new Blob([videoFile], { type: videoFile.type });

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

        // ✅ Now that currentVideoId is available, save the blob
        await saveVideoBlobToIndexedDB(blob, currentVideoId);

        const db = await initDB();
        const tx = db.transaction('videos', 'readonly');
        const store = tx.objectStore('videos');
        const check = store.get(currentVideoId);

        check.onsuccess = () => {
        const result = check.result;
        if (result) {
            console.log("🧩 SUCCESS: Video was saved and can be read back");
        } else {
            console.warn("❌ SAVING FAILED: Key not found in DB");
        }
        };
        check.onerror = () => console.error("❌ Error while verifying DB write");


        console.log("✅ Saved video to IndexedDB with key", currentVideoId);

        pollTaskStatus2(data.task_id, data.audio_url);

    } catch (error) {
        console.error('Error during video upload:', error);
        updateStatus('Error Uploading');

        if (videoElement) videoElement.style.opacity = '1';
        if (loaderOverlay) loaderOverlay.style.display = 'none';
        loading.style.display = 'none';

        document.querySelector('#genmus').style.pointerEvents = 'auto';
        document.querySelector('#genmus').style.opacity = '100%';
        document.querySelector('#gensfx').style.pointerEvents = 'auto';
        document.querySelector('#gensfx').style.opacity = '100%';
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

let sceneChanges = []

function pollTaskStatus2(taskId, audioUrl) {
  const statusInterval = setInterval(() => {
    fetch(`https://doseedo.com/task-status/${taskId}`)
      .then(response => response.json())
      .then(data => {
        updateStatus(data.status);

        if (data.status === 'SUCCESS') {
          clearInterval(statusInterval);
            const videoElement = document.getElementById('player');
            videoElement.style.opacity = '1';
            document.getElementById('videoLoader').style.display = 'none';

          // UI Setup
        //   document.getElementById('videoLoader')?.style?.display = 'none';
         
          if (videoElement) {
            videoElement.muted = true;
            videoElement.style.opacity = '1';
            originalLengthInSeconds = videoElement.duration;
            totalDuration = videoElement.duration;
            zoomSlider.value = "1";
            zoomSlider.dispatchEvent(new Event('input'));
          }

          const sceneData = data.result.scene_data;
          const sceneChanges = data.result.scene_changes || [];
          const sceneDurations = sceneChanges.slice(1).map((t, i) => t - sceneChanges[i]);

          localStorage.setItem("sceneData", JSON.stringify(sceneData));
          localStorage.setItem("sceneDurations", JSON.stringify(sceneDurations));

          // Collapse short scenes
          const collapseSceneChanges = (changes, threshold) => {
            if (!changes.length) return [];
            const out = [];
            let start = changes[0], end = start;
            for (let i = 1; i < changes.length; i++) {
              const t = changes[i];
              if (t - end < threshold) {
                end = t;
              } else {
                out.push(start);
                if (end !== start) out.push(end);
                start = end = t;
              }
            }
            if (start === end) out.push(start);
            else { out.push(start); out.push(end); }
            return out;
          };

          const collapsed = collapseSceneChanges(sceneChanges, 3);
          window.originalSceneChanges = [...collapsed];




          document.getElementById('muss').classList.add('selected');

          // Build per-scene summary
          const sceneSummaries = sceneData.map((scene, i) => {
            const labels = scene.key_frames.flatMap(k => k.analysis.labels?.map(l => l.description) || []);
            const texts = scene.key_frames.flatMap(k => k.analysis.text?.map(t => t.content) || []);
            const objects = scene.key_frames.flatMap(k => Object.keys(k.analysis.objects || {}));
            const summary = [
              [...new Set(labels)].join(', '),
              [...new Set(objects)].join(', '),
              [...new Set(texts)].slice(0, 3).join(' | ')  // Limit text
            ].filter(Boolean).join(' | ');
            return `Scene ${i + 1}: ${summary}`;
          });

          const descriptionInput = document.getElementById('description-input');
        //   descriptionInput.value = sceneSummaries.join('\n\n');

          // Automation points
          globalSceneChanges = collapsed;
          renderSceneRangeBars(globalSceneChanges);

          const leftEdge = automationPoints.find(p => p.isEdge && p.time === 0);
          const midY = leftEdge?.y ?? volumeToY(0.5);
          globalSceneChanges.forEach(time => {
            automationPoints.push({ time, y: midY, isEdge: false });
          });
          updateAutomationData();
          draw();

          resizeCanvas();
          draw();
          renderTimeline(originalLengthInSeconds, currentZoomLevel, document.getElementById('download-links').offsetWidth);

          // Enable buttons
          loading.style.display = 'none';
          document.querySelector('#genmus').style.pointerEvents = 'auto';
          document.querySelector('#genmus').style.opacity = '100%';
          document.querySelector('#gensfx').style.pointerEvents = 'auto';
          document.querySelector('#gensfx').style.opacity = '100%';

          // Optional audio waveform
          const serverAudioUrl = data.result.audio_url;
          if (serverAudioUrl) {
            createDownloadLinks([`https://doseedo.com${serverAudioUrl}`], sceneChanges, 'download-links3', false, true);
          }

          // Cue system
          if (sceneChanges.length > 0) {
            triggerCueingSystem(sceneChanges, []);
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
        loading.style.display = 'none';
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

function renderSceneRangeBars(sceneChanges) {
    const timeline = document.getElementById('scene-bar-overlay');
    if (!timeline) return;
    timeline.innerHTML = '';

    const containerWidth = document.getElementById('download-links').offsetWidth * currentZoomLevel;

    const pixelsPerSecond = getPixelsPerSecond();


    for (let i = 0; i < sceneChanges.length; i++) {
        const startTime = sceneChanges[i];
        const endTime = sceneChanges[i + 1] ?? originalLengthInSeconds;
        const startX = startTime * pixelsPerSecond;
        const width = (endTime - startTime) * pixelsPerSecond;

        const bar = document.createElement('div');
        bar.className = 'scene-range draggable-scene-marker';
        bar.dataset.index = i;
        bar.style.left = `${startX}px`;
        bar.style.width = `${width}px`;
        bar.title = `Scene ${i + 1}`;
        // bar.style.top = `${0-(i*2)}px`;

        // pastel pink → purple → blue
        const r = Math.max(200 - i * 10, 100);
        const g = Math.max(200 - i * 20, 100);
        const b = Math.min(200 + i * 25, 255);
        bar.style.background = `rgba(${r}, ${g}, ${b}, 0.8)`;

            // Create arrow at front of scene
        const arrow = document.createElement('div');
        arrow.className = 'scene-arrow';
        arrow.style.left = `${startX}px`;
        arrow.style.top = '-12px'; // visually above the bar
        arrow.title = `Scene ${i + 1}`;

    timeline.appendChild(arrow);
        timeline.appendChild(bar);



        // ✅ Attach interact.js draggable handler
        interact(bar).draggable({
            axis: 'x',
            listeners: {
                move(event) {
                    const target = event.target;
                    const dx = event.dx;

                    const containerWidth = document.getElementById('timeline-bar').offsetWidth * currentZoomLevel;
                    const pixelsPerSecond = getPixelsPerSecond();


                    const index = parseInt(target.dataset.index);
                    if (isNaN(index)) return;

                    const currentLeft = parseFloat(target.style.left);
                    const newLeft = currentLeft + dx;
                    target.style.left = `${newLeft}px`;

                    const newTime = newLeft / pixelsPerSecond;

                    const min = index === 0 ? 0 : globalSceneChanges[index - 1] + 0.1;
                    const max = index === globalSceneChanges.length - 1
                        ? originalLengthInSeconds
                        : globalSceneChanges[index + 1] - 0.1;

                    globalSceneChanges[index] = Math.min(Math.max(newTime, min), max);
                },
                end() {
                    const containerWidth = document.getElementById('download-links').offsetWidth * currentZoomLevel;
                    // const pixelsPerSecond = containerWidth / originalLengthInSeconds;

                    renderSceneRangeBars(globalSceneChanges);
                    renderTimeline(originalLengthInSeconds, currentZoomLevel, containerWidth);
                
                                        // 🎯 Load cues
                    

                }
            }
        });
    }
}




function triggerCueingSystem(sceneChanges, detectedImpacts, audioFilesFromBackend = []) {



    sceneChanges.forEach(time => {
        console.log(`Trigger scene change cue at ${time} seconds.`);
        const cue = renderCueArrow(time, 'Scene Change', 'scene');
        renderSceneRangeBars(globalSceneChanges);
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
        // createDownloadLinks(audioFiles, sceneTiming);
        // console.log('asouhfsaou')
    }
}






const timelineBar = document.getElementById('timeline-bar');
const downloads = document.querySelector('.downloads');

if (window.location.pathname == '/doseedo1.html') {


downloads.addEventListener('click', function (e) {
    const rect = timelineBar.getBoundingClientRect();
    const clickX = e.clientX - rect.left;

    const containerWidth = document.getElementById('download-links').offsetWidth * currentZoomLevel;
    
    const pixelsPerSecond = getPixelsPerSecond();


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

const SNAP_X_THRESHOLD = 0.3; // seconds

document.querySelector('.downloadbox').addEventListener('mousemove', (e) => {
    const downloadBox = document.querySelector('.downloadbox');
    const timelineBar = document.getElementById('timeline-bar');
    const rect = downloadBox.getBoundingClientRect();
    
    const relativeX = e.clientX - rect.left;  // X inside .downloadbox
    const pixelsPerSecond = getPixelsPerSecond();
    const snappedTime = relativeX / pixelsPerSecond;

    // Position the cursor inside the timeline, which scrolls with .downloadbox
    const snappedX = snappedTime * pixelsPerSecond;

    cursorLine.style.transform = `translateX(${snappedX}px)`;  // No extra offset
    cursorTime.innerText = snappedTime.toFixed(2) + 's';
    cursorTime.style.left = `${snappedX + 5}px`;
    cursorLine.style.display = 'block';
    cursorTime.style.display = 'block';
});




document.querySelector('.downloadbox').addEventListener('mouseleave', () => {
    cursorLine.style.display = 'none';
    cursorTime.style.display = 'none';
});


}



let newbatch = false


function pollTaskStatus(taskId, callback) {
    fetch(`https://doseedo.com/task/${taskId}`)
        .then(response => response.json())
        .then(data => {
            if (!data || !data.status) {
                console.log('PENDING');
                setTimeout(() => pollTaskStatus(taskId, callback), 2000);
                return;
            }

            if (data.status === 'completed') {
                const fullPaths = (data.file_paths || data.result || []).map(path => `https://doseedo.com${path}`);

                // Only overwrite if sequence_groups exists and original scene timings are present
                if (
                    Array.isArray(data.sequence_groups) &&
                    data.sequence_groups.length &&
                    Array.isArray(globalSceneChanges) &&
                    globalSceneChanges.length
                ) {
                    const originalSceneChanges = [...globalSceneChanges];

                    globalSceneChanges = [];

                    for (const group of data.sequence_groups) {
                        const start = originalSceneChanges[group[0] - 1];
                        globalSceneChanges.push(start);
                    }

                    const lastGroup = data.sequence_groups[data.sequence_groups.length - 1];
                    const finalEnd = originalSceneChanges[lastGroup[lastGroup.length - 1]];
                    globalSceneChanges.push(finalEnd);
                }

                // Fallback if still missing
                if (!globalSceneChanges || !globalSceneChanges.length) {
                    console.warn("⚠️ globalSceneChanges not ready, falling back to dummy timings");
                    globalSceneChanges = fullPaths.map((_, i) => i * 2);
                }

                // ✅ Restored behavior
                createDownloadLinks(fullPaths, globalSceneChanges);
                newbatch = true;
                if (callback) callback(fullPaths);

                updateStatus('Task completed. Download available.');
                document.querySelector('#genmus').style.pointerEvents = 'auto';
                document.querySelector('#genmus').style.opacity = '1';
                document.querySelector('#gensfx').style.opacity = '1';
                document.querySelector('#gensfx').style.pointerEvents = 'auto';
                loading.style.display = 'none';
            } else {
                updateStatus(`Processing... (${data.status})`);
                setTimeout(() => pollTaskStatus(taskId, callback), 2000);
            }
        })
        .catch(error => {
            console.error('Error polling task status:', error);
            updateStatus('Error during task status polling.');
            loading.style.display = 'none';
        });
}





let trackStateMap = new Map(); // map of original positions




function setupGlobalControls() {

    const downloadsContainer = document.querySelector('.downloadbox');
    const footer = document.querySelector('#footer');

    // document.querySelectorAll('#undo')[0].addEventListener('click', undo);
    // document.querySelectorAll('#undo')[1].addEventListener('click', redo);

    const containerWidth = document.getElementById('download-links').offsetWidth;

        // 🧠 Toggle overflow mode on .downloads
        const downloadsDiv = document.querySelector('.downloads');
    if (downloadsDiv) {
        if (currentZoomLevel === 1) {
            downloadsDiv.style.overflowX = 'hidden';  // lock
            downloadsDiv.scrollLeft = 0;              // reset to far left
        } else {
            downloadsDiv.style.overflowX = 'scroll';  // allow manual scroll
        }
    }

        // const containerWidth = 1000; // Fixed base width

    renderTimeline(originalLengthInSeconds, currentZoomLevel, containerWidth);

    // Prevent duplicate creation
    if (document.getElementById('slider-container')) return;

;



 

    const sliderContainer = document.createElement("div")
    // sliderContainer.style.marginBottom = "10px";
    // sliderContainer.style.display = "flex";
    sliderContainer.style.alignItems = "center";
    sliderContainer.style.gap = "10px";
    sliderContainer.id = "slider-container";

    const heightSlider = document.createElement("input");
    heightSlider.type = "range";
    heightSlider.min = "30";
    heightSlider.max = "150";
    heightSlider.value = "60";
    heightSlider.style.width = "10%";
    heightSlider.id = "height-slider"; // assign ID to access later

    let zoomSlider = document.createElement("input");
    zoomSlider.type = "range";
    zoomSlider.min = "1";
    zoomSlider.max = "10";
    zoomSlider.step = "0.1";
    zoomSlider.style.width = "10%";

    zoomSlider.value = "1";
    zoomSlider.id = "zoom-slider"; // assign ID to access later

    zoomSlider.addEventListener('input', () => {
        currentZoomLevel = parseFloat(zoomSlider.value);
        const containerWidth = document.getElementById('download-links').offsetWidth;

            // 🧠 Toggle overflow mode on .downloads
            const downloadsDiv = document.querySelector('.downloads');
    if (downloadsDiv) {
        if (currentZoomLevel === 1) {
            downloadsDiv.style.overflowX = 'hidden';  // lock
            downloadsDiv.scrollLeft = 0;              // reset to far left
        } else {
            downloadsDiv.style.overflowX = 'scroll';  // allow manual scroll
        }
    }

        // const containerWidth = 1000; // Fixed base width

        resizeCanvas();
        draw();
        renderTimeline(originalLengthInSeconds, currentZoomLevel, containerWidth);
        document.querySelectorAll('#download-links li, #download-links2 li, #download-links3 li').forEach(updateTrackPosition);
    });

    


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
    heightSlider.style.display = 'none'
    sliderContainer.appendChild(zoomSlider);

    sliderContainer.appendChild(deleteButton);
    sliderContainer.appendChild(syncbutton);


        // insert it right AFTER timeline-wrapper

        const timelineWrapper = document.querySelector('.timeline-wrapper');


    // timelineWrapper.insertAdjacentElement('beforebegin', sliderContainer);
    footer.appendChild(sliderContainer);

    draw()
    resizeCanvas();

    
}


if (window.location.pathname == '/doseedo1.html' ) {

document.getElementById('sync-toggle').addEventListener('change', (e) => {
    syncToSceneChanges = e.target.checked;
});

let zoomSlider = document.getElementById("zoom-slider");

}

// 1. Proper decodeAndAttachBuffer function
async function decodeAndAttachBuffer(track, fileUrl) {
    try {
        const ctx = initAudioContext();
        let arrayBuffer;

        if (fileUrl instanceof File) {
            arrayBuffer = await fileUrl.arrayBuffer();
        } else if (typeof fileUrl === 'string' && fileUrl.startsWith('http')) {
            const response = await fetch(fileUrl);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            arrayBuffer = await response.arrayBuffer();
        } else {
            throw new Error('Invalid file URL type');
        }

        // Single decode operation
        track._decodedBuffer = await new Promise((resolve, reject) => {
            ctx.decodeAudioData(arrayBuffer, resolve, reject);
        });
        
        console.log('✅ Decoded and attached buffer');
    } catch (err) {
        console.error('❌ Audio processing failed:', err);
        throw err; // Re-throw for error handling upstream
    }
}

// 2. Track metadata initialization - Typically in your track creation code
function createNewTrackElement() {
    const listItem = document.createElement("li");
    
    // Initialize metadata with all required properties
    listItem._meta = {

        source: '',
        startPositionInSeconds: 0,
        trackDuration: 0,
        trimStart: 0,
        trimEnd: 0,
        fadeOutDuration: parseFloat(fadeOutSlider.value) || 1.0,
        fadeInDuration: parseFloat(fadeInSlider.value) || 0.2,
        fadeToEndLine: fadeToEndCheckbox.checked,
        reverbAmount: 0.2,
        isInitiallyLoaded: false, // ← Add this flag here
        get endPositionInSeconds() {
            return this.startPositionInSeconds + this.trackDuration;
        }
    };

    // Rest of track initialization...
    return listItem;
}

// 3. Usage in your track creation flow
async function createDownloadLinks(filePaths) {
    filePaths.forEach(async (filePath) => {
        const track = createNewTrackElement();
        track._meta.source = filePath;
        
        try {
            await decodeAndAttachBuffer(track, filePath);
            // Successful decode, track is ready for controlled playback
        } catch (error) {
            console.error('Failed to process track:', error);
            track.remove();
        }
    });
}








function createDownloadLinks(filePaths, sceneChanges = [], containerId = 'download-links', isRegen = false) {
    if (!audioContext) {
        audioContext = monitorAudioContext();
        document.body.addEventListener('click', () => {
            if (audioContext.state === 'suspended') {
                audioContext.resume();
            }
        }, { once: true });
    }
    
    let downloadList = document.getElementById('download-links');
    downloadList = document.getElementById(containerId);
    const validSceneChanges = Array.isArray(globalSceneChanges) 
    ? globalSceneChanges 
    : [];
    
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
    const sceneCount = filePaths.length;

    if (!Array.isArray(sceneChanges) || sceneChanges.length !== sceneCount + 1) {
        if (Array.isArray(globalSceneChanges) && globalSceneChanges.length === sceneCount + 1) {
            console.warn("🎯 Using globalSceneChanges for sequence group timings");
            sceneChanges = globalSceneChanges;
        } else {
            console.warn("⚠️ No valid sceneChanges or globalSceneChanges — using fallback");
            sceneChanges = Array.from({ length: sceneCount }, (_, i) => i * 2);
            sceneChanges.push(sceneChanges[sceneCount - 1] + 2);
        }
    }

    

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

    if (!window.trackAppendMode && typeof window._regenReplaceIndexInDOM !== 'number') {
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

  

    // zoomSlider.removeEventListener('input', applyZoom); // avoid duplicates
    // zoomSlider.addEventListener('input', applyZoom);
;




    



function deleteSelectedTracks() {
    const selectedWaveforms = document.querySelectorAll('.waveform-container.selected');

    if (selectedWaveforms.length === 0) {
        alert("No track selected to delete!");
        return;
    }

    selectedWaveforms.forEach(waveform => {
        const li = waveform.closest('li');
        const container = li.closest('[id^="download-links"]');
        const index = [...container.children].indexOf(li);

        // Remove matching label
        let tracklist;
        if (container.id === 'download-links') tracklist = document.querySelector('#muss .tracklist');
        else if (container.id === 'download-links2') tracklist = document.querySelector('#sfxs .tracklist');
        else if (container.id === 'download-links3') tracklist = document.querySelector('#vos .tracklist');

        if (tracklist && tracklist.children[index]) {
            tracklist.children[index].remove();
        }

        li.remove();
    });

    // Reflow and relabel remaining tracks in all sections
    ['download-links', 'download-links2', 'download-links3'].forEach(containerId => {
        const container = document.getElementById(containerId);
        const tracklist = container.closest('.downloadbox').querySelector('.tracklist');
        const lis = container.querySelectorAll('li');

        lis.forEach((li, i) => {
            const newTop = i * 60;
            li.style.top = `${newTop}px`;
            li.dataset.originalTop = newTop;
            li.style.zIndex = (3000 - (i * 100));

            // Update label numbers
            const label = tracklist.children[i];
            if (label) {
                const labelName = label.innerText.split(': ').slice(1).join(': ');
                label.innerText = `${i + 1}: ${labelName}`;
            }
        });
    });
}

// Hook up the delete button
deleteButton.addEventListener("click", deleteSelectedTracks);

// Hook up the keyboard Delete key
document.addEventListener('keydown', (e) => {
    if ((e.key === 'Delete' || e.key === 'Backspace') && document.querySelector('.waveform-container.selected')) {
        deleteSelectedTracks();
        e.preventDefault();
    }
});












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

    let injectAtIndex = window._regenInsertIndex ?? null;
    let injectAtTop = window._regenInsertTop ?? null;
    let targetContainerId = window._regenContainerId ?? containerId;

    // Cleanup
    delete window._regenInsertIndex;
    delete window._regenInsertTop;
    delete window._regenContainerId;

    let injectDOMIndex = window._regenReplaceIndexInDOM ?? null;
    delete window._regenReplaceIndexInDOM;


    // Replace downloadList with target if overridden
    if (targetContainerId !== containerId) {
        downloadList = document.getElementById(targetContainerId);
    }


    filePaths.forEach((filePath, index) => {
        

        const audioLink = filePath;



        const listItem = document.createElement("li");
        listItem.dataset.uuid = crypto.randomUUID(); // needs to persist across edits


        let indexToUse = typeof injectAtIndex === 'number' ? injectAtIndex : currentLiCount;
        let topToUse = typeof injectAtTop === 'number' ? injectAtTop : indexToUse * 60;

        listItem.dataset.originalIndex = indexToUse;
        listItem.dataset.originalTop = topToUse;

        
        listItem.dataset.sceneIndex = injectAtIndex ?? index;
        listItem.style.top = `${topToUse}px`;

        // Only increment counter if we're in normal append mode
        if (injectAtIndex === undefined || injectAtIndex === null) {
            currentLiCount++;
        }


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

        const maxLabelLength = 10; // change as needed
        const truncatedLabel = labelName.length > maxLabelLength
        ? labelName.slice(0, maxLabelLength) + '…'
        : labelName;

        trackLabel.innerText = `${index + 1}: ${truncatedLabel}`;
        trackLabel.dataset.index = listItem.dataset.originalIndex;
        trackLabel.classList.add("tracklabel");



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
        soloButton.classList = 'soloButton'



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
        
        
        if (!isRegen) {
            tracklistForThisDownloadList?.appendChild(labelContainer);
        }
        
        



        const waveformContainer = document.createElement("div");
        waveformContainer.className = 'waveform-container';
        waveformContainer.style.position = 'absolute';
        waveformContainer.style.height = '50px';
        // waveformContainer.style.overflow = 'hidden';
        // waveformContainer.style.border = '1px solid #000';

        const leftMask = document.createElement('div');
        leftMask.className = 'waveform-mask-left';
        waveformContainer.appendChild(leftMask);
        listItem._leftMask = leftMask;


        const rightMask = document.createElement('div');
        rightMask.className = 'waveform-mask-right';
        waveformContainer.appendChild(rightMask);
        listItem._rightMask = rightMask;

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
        
        if (typeof injectDOMIndex === 'number') {
            const existingItems = Array.from(downloadList.children);
            const adjustedIndex = Math.min(injectDOMIndex, existingItems.length);
        
            const oldTrack = window._regenReplaceElement;

            if (oldTrack && oldTrack.parentElement === downloadList) {
                downloadList.replaceChild(listItem, oldTrack);  

                const existingGif = oldTrack.querySelector('.regen-loading-indicator');
                if (existingGif) existingGif.remove();
                
                // ✅ Replace in place
            } else {
                downloadList.appendChild(listItem);             // Fallback
            }
            
            delete window._regenReplaceElement;


        } else {
            downloadList.appendChild(listItem);
            if (!isRegen) {
                tracklistForThisDownloadList?.appendChild(labelContainer);
            }
        }
        
        
        

        
        // ✅ Recalculate tops so everything is spaced properly
        
        ['download-links', 'download-links2', 'download-links3'].forEach(containerId => {
            const container = document.getElementById(containerId);
            const trackItems = container.querySelectorAll('li');
          
            trackItems.forEach((li, i) => {
              const top = i * 60;
              li.style.top = `${top}px`;
              li.dataset.originalTop = top;
              li.style.zIndex = i * 100;
            });
          });
          



        
        // ✅ Return to collapsed state if it started that way
        // if (isCollapsed) {
        //   setTimeout(() => {
        //     downloadList.classList.remove('expanded');
        //     // Flatten all tops after collapse
        //     allTrackItems.forEach(li => {
        //       li.style.top = '0px';
        //     });
        //   }, 50);
        // }

        



        
        let startPositionInSeconds;
        if (typeof injectAtIndex === 'number') {
            startPositionInSeconds = globalSceneChanges[injectAtIndex] ?? injectAtIndex * 2;
        } else {
            startPositionInSeconds = sceneChanges[index] ?? index * 2;
        }

        
        let nextStart = sceneChanges[index + 1];
        let defaultDuration = isFullDuration
            ? originalLengthInSeconds - startPositionInSeconds
            : (nextStart !== undefined
                ? nextStart - startPositionInSeconds
                : originalLengthInSeconds - startPositionInSeconds);

        let trackDuration = Math.max(0.1, defaultDuration);

        // ✅ assign to _meta
        if (!listItem._meta) listItem._meta = {};
        listItem._meta.trackDuration = trackDuration;
        listItem._meta.trimEnd = trackDuration;




        



        const rand = Math.floor(Math.random() * 50);
        
        const containerWidth = downloadList.offsetWidth * currentZoomLevel;
        const pixelsPerSecond = getPixelsPerSecond();

        const pixelWidth = trackDuration * pixelsPerSecond;

        waveformContainer.style.width = `${pixelWidth}px`;



        const ctx = initAudioContext();



        const convolver = ctx.createConvolver();
        convolver.buffer = reverbIRBuffer;

        if (reverbIRBuffer) {
            convolver.buffer = reverbIRBuffer;
        } else {
            console.warn('Reverb impulse response not loaded');
        }


        const gainNode = ctx.createGain();
        gainNode.gain.value = 1.0;  // default volume

        const reverbGain = ctx.createGain();
        reverbGain.gain.value = 0.2; // default reverb amount

        

        // Chain: dry gain + wet reverb path → output
        gainNode.connect(ctx.destination);
        reverbGain.connect(ctx.destination);

        const wavesurfer = WaveSurfer.create({
            // backend: 'WebAudio',
            container: waveformContainer,
            loadMethod: 'loadBlob',
            audioContext: ctx,
            responsive: true,
            // normalize: false,
            waveColor: 'white',
            progressColor: 'white',
            barWidth: 2,
            // height: parseInt(heightSlider.value),
            height: 60,
            backend: 'WebAudio',
            renderer: 'MultiCanvas',
            normalize: false,
            splitChannels: false,
            // autoCenter: false,
            partialRender: true,
            autoplay: false, // Ensure autoplay is disabled
            interact: false, 
        });



// wavesurfer.on('play', () => {
//     const ctx = initAudioContext();
//     const backend = wavesurfer.backend;
//     const source = backend.bufferSource;
    
//     // Create fresh audio nodes with current values
//     const dryGain = ctx.createGain();
//     const convolver = ctx.createConvolver();
//     const wetGain = ctx.createGain();
//     const masterGain = ctx.createGain();

//     // Initialize with current metadata values
//     dryGain.gain.value = 1 - track._meta.reverbAmount;
//     wetGain.gain.value = track._meta.reverbAmount;
    
//     // Apply fade-out if needed
//     if (track._meta.fadeOutDuration > 0) {
//         const now = ctx.currentTime;
//         masterGain.gain.setValueAtTime(1, now);
//         masterGain.gain.linearRampToValueAtTime(0, now + track._meta.fadeOutDuration);
//     }

//     // Connect nodes
//     source.connect(dryGain);
//     source.connect(convolver);
//     convolver.connect(wetGain);
//     dryGain.connect(masterGain);
//     wetGain.connect(masterGain);
//     masterGain.connect(ctx.destination);

//     // Store nodes for real-time control
//     track._fx = { dryGain, wetGain, masterGain, convolver };
// });




    
        if (audioLink instanceof File) {
            wavesurfer.loadBlob(audioLink);
        } else if (typeof audioLink === 'string' && audioLink.startsWith('http')) {
            fetch(audioLink)
                .then(r => r.blob())
                .then(blob => wavesurfer.loadBlob(blob))
                .catch(err => console.error("❌ Blob load failed", err));
        } else {
            console.warn("⚠️ Skipping fetch: invalid or local-only file name:", audioLink);
        }

        
        
        


        decodeAndAttachBuffer(listItem, audioLink);
        



        listItem._wavesurfer = wavesurfer;


        // 🔊 Custom manual playback with effects
// 🔊 Custom manual playback with effects
// Modified playWithEffects function
function playWithEffects(track) {
    if (!track?._meta || !track._decodedBuffer) return;

    const ctx = initAudioContext();

    // 🔇 Stop and mute WaveSurfer to prevent overlap
    track._wavesurfer?.pause();
    track._wavesurfer?.setVolume(0);

    // 🧼 Stop and disconnect any old FX nodes
    if (track._fx) {
        try { track._fx.source?.stop(); } catch (e) {}
        track._fx.source?.disconnect();
        track._fx.dryGain?.disconnect();
        track._fx.wetGain?.disconnect();
        track._fx.convolver?.disconnect();
        track._fx.masterGain?.disconnect();
    }

    const source = ctx.createBufferSource();
    const dryGain = ctx.createGain();
    const wetGain = ctx.createGain();
    const convolver = ctx.createConvolver();
    const masterGain = ctx.createGain();

    source.buffer = track._decodedBuffer;
    convolver.buffer = reverbIRBuffer;

    const meta = track._meta;
    const reverbAmt = meta.reverbAmount;
    const fadeOutAmt = meta.fadeOutDuration || 0;
    const fadeInAmt = meta.fadeInDuration || 0;

    dryGain.gain.value = 1 - reverbAmt;
    wetGain.gain.value = reverbAmt;

    source.connect(dryGain);
    source.connect(convolver);
    convolver.connect(wetGain);
    dryGain.connect(masterGain);
    wetGain.connect(masterGain);

    const isMuted = track.classList.contains('muted');
    const isSoloActive = !!document.querySelector('.soloed');

    // 💡 Mute if muted OR if some solo is active and this track isn't soloed
    const shouldMute = isMuted || (isSoloActive && !track.querySelector('.soloed'));


    if (!shouldMute) {
        masterGain.connect(ctx.destination);
    } else {
        console.log("🔇 Skipping playback for muted track");
        return;
    }

    const currentVideoTime = videoPreviewsrc.currentTime;
    const trackStart = meta.startPositionInSeconds + meta.trimStart;
    const trackEnd = meta.startPositionInSeconds + meta.trimEnd;

    // Calculate offset and duration
    
    // If fade is set to go beyond trimEnd, allow extra duration
    const extraTail = (!meta.fadeToEndLine && meta.fadeOutDuration) || 0;
    const playbackTrimEnd = meta.fadeToEndLine
        ? meta.trimEnd
        : Math.min(meta.trackDuration + extraTail, track._decodedBuffer.duration);

    const duration = Math.min(
        playbackTrimEnd - meta.trimStart - offset,
        track._decodedBuffer.duration - meta.trimStart
    );



    const now = ctx.currentTime;

    // 🎚 Fade-in
    if (fadeInAmt > 0.01) {
        masterGain.gain.setValueAtTime(0.001, now);
        masterGain.gain.exponentialRampToValueAtTime(1.0, now + fadeInAmt);
    } else {
        masterGain.gain.setValueAtTime(1.0, now);
    }

    // 🎚 Fade-out
    const fadeStart = trackEnd - fadeOutAmt;
    const timeUntilFade = fadeStart - currentVideoTime;

    if (fadeOutAmt > 0.01) {
        if (timeUntilFade > 0) {
            masterGain.gain.setValueAtTime(1.0, now + timeUntilFade);
            masterGain.gain.exponentialRampToValueAtTime(0.001, now + timeUntilFade + fadeOutAmt);
        } else {
            const elapsed = currentVideoTime - fadeStart;
            const remaining = Math.max(0, fadeOutAmt - elapsed);
            const initialGain = Math.max(0.001, 1 - (elapsed / fadeOutAmt));

            masterGain.gain.setValueAtTime(initialGain, now);
            if (remaining > 0.01) {
                masterGain.gain.exponentialRampToValueAtTime(0.001, now + remaining);
            }
        }
    }

    source.start(now, meta.trimStart + offset, duration);

    track._fx = { source, dryGain, wetGain, masterGain, convolver };
    track._meta.reverbAmount = reverbAmt;
    track._meta.fadeOutDuration = fadeOutAmt;
    track._meta.fadeInDuration = fadeInAmt;
}


// Simplified Reverb Handler
const handleReverbChange = debounce(() => {
    const track = getSelectedTrack();
    if (!track?._fx) return;

    const amt = Math.min(1, Math.max(0, parseFloat(reverbSlider.value)));
    
    // Dry signal remains constant at 1, only adjust wet
    track._fx.wetGain.gain.value = amt;
    track._meta.reverbAmount = amt;
}, 50);

// Remove duplicate reverb listeners
reverbSlider.removeEventListener('input', handleReverbChange);
reverbSlider.addEventListener('input', handleReverbChange);



playButton.addEventListener("click", async () => {
    try {
        const ctx = initAudioContext();
        
        // Required by browser autoplay policies - must be in click handler
        if (ctx.state === 'suspended') {
            await ctx.resume();
        }

        const video = document.getElementById('player');
        const meta = listItem._meta;
        const startDelay = meta.startPositionInSeconds + meta.trimStart;

        if (listItem.classList.contains('muted')) {
            alert("Track is muted - unmute to play");
            return;
        }

        // Clear existing audio nodes
        if (listItem._fx?.source) {
            listItem._fx.source.stop();
            listItem._fx = null;
        }

        // Immediate check first
        if (video.currentTime >= startDelay) {
            playWithEffects(listItem);
        } else {
            // Wrap in a single RAF callback
            const waitForStart = () => {
                if (video.currentTime >= startDelay) {
                    listItem._wavesurfer.setVolume(0);
                    listItem._wavesurfer.pause();
                    playWithEffects(listItem);
                    playButton.innerText = "⏸";
                } else {
                    requestAnimationFrame(waitForStart);
                }
            };
            requestAnimationFrame(waitForStart);
        }

        // Visual feedback
        playButton.innerText = "⏸";
    } catch (error) {
        console.error('Playback error:', error);
        alert('Audio playback failed - please click again');
    }
});

        
        
        





        
        listItem._meta = {
            source: audioLink, 
            startPositionInSeconds,
            trackDuration,
            trimStart: 0,
            fadeOutDuration: parseFloat(fadeOutSlider.value) || 1.0,
            fadeInDuration: parseFloat(fadeInSlider.value) || 0.2,
            reverbAmount: 0.2,              // ✅ initialize here
            get endPositionInSeconds() {
                return this.startPositionInSeconds + this.trackDuration;
            }
        };

        // Calculate pre-render pixel width using known scene duration and zoom
        // const containerWidth = downloadList.offsetWidth * currentZoomLevel;
        // const pixelsPerSecond = containerWidth / originalLengthInSeconds;
        const visibleDuration = trackDuration;  // You already defined this
        const preWaveformWidth = visibleDuration * pixelsPerSecond;

        // Set early before WaveSurfer renders anything
        waveformContainer.style.width = `${preWaveformWidth}px`;



        Promise.all([
  decodeAndAttachBuffer(listItem, audioLink),
  new Promise(resolve => wavesurfer.on('ready', resolve))
]).then(() => {
  const ctx = initAudioContext();
  const buffer = listItem._decodedBuffer;

  if (!(buffer instanceof AudioBuffer)) {
    console.warn("❌ Still no decoded buffer");
    return;
  }

  // Setup effect chain
  const source = ctx.createBufferSource();
  source.buffer = buffer;

  const dryGain = ctx.createGain();
  const convolver = ctx.createConvolver();
  const wetGain = ctx.createGain();
  const masterGain = ctx.createGain();

  const amt = listItem._meta?.reverbAmount ?? 0.2;
  dryGain.gain.value = 1 - amt;
  wetGain.gain.value = amt;
  convolver.buffer = reverbIRBuffer;

  source.connect(dryGain).connect(masterGain);
  source.connect(convolver).connect(wetGain).connect(masterGain);
  masterGain.connect(ctx.destination);

  source.start();
source.stop();

  listItem._fx = { dryGain, wetGain, masterGain, convolver, source };

  // Optional: visual scaling
  const maxAmplitude = buffer.getChannelData(0)
    .reduce((max, val) => Math.max(max, Math.abs(val)), 0);
  const volumeScale = Math.max(0.2, maxAmplitude);

//   waveformContainer.style.transform = `scaleY(${volumeScale})`;
//   waveformContainer.style.transformOrigin = 'bottom';

const canvas = waveformContainer.querySelector('canvas');
if (canvas && buffer instanceof AudioBuffer) {
  const maxAmp = buffer.getChannelData(0)
    .reduce((max, val) => Math.max(max, Math.abs(val)), 0);

  const scaleMultiplier = 6;                // make loudness pop
  const volumeScale = Math.min(3.0, Math.max(0.2, maxAmp * scaleMultiplier));

  // 🧠 Combine centering with vertical scaling
  canvas.style.transform = `translateY(-50%) scaleY(${volumeScale})`;
  canvas.style.transformOrigin = 'center';
}



  // WaveSurfer sync setup
  console.log("✅ WaveSurfer is ready");
  console.log("⏱ Full Duration:", wavesurfer.getDuration());

  wavesurfer.isReady = true;
  listItem.dataset.ready = 'true';

  const meta = listItem._meta;
  const fullDuration = wavesurfer.getDuration();
  meta.trackDuration = fullDuration;

  if (typeof meta.trimStart !== 'number') {
        meta.trimStart = 0;
    }
    if (typeof meta.trimEnd !== 'number') {
    const sceneIndex = parseInt(listItem.dataset.sceneIndex, 10);
    const nextSceneStart = globalSceneChanges[sceneIndex + 1] ?? originalLengthInSeconds;
    const sceneTrim = nextSceneStart - meta.startPositionInSeconds;

    // ❗ Don't clamp risers (download-links2)
    const isRiser = downloadList.id === 'download-links2';
    meta.trimEnd = isRiser
        ? fullDuration
        : Math.min(fullDuration, sceneTrim);
    }


  if (wavesurfer.drawer?.container) {
    wavesurfer.drawer.container.style.width = `${pixelWidth}px`;
    wavesurfer.drawer.progress(0);
  }

  wavesurfer.setOptions({
    minPxPerSec: pixelsPerSecond,
    autoCenter: false,
    interact: false,
    hideScrollbar: true
  });

  wavesurfer.zoom(pixelsPerSecond);
  requestAnimationFrame(() => updateTrackPosition(listItem));
});




        
        
        


        // interact(waveformContainer).resizable({
        //     edges: { left: true, right: true },
        //     listeners: {
        //         start(event) {
        //             if (!selectedTracks.includes(listItem)) {
        //                 selectedTracks.forEach(el => el.classList.remove('selected'));
        //                 selectedTracks = [listItem];
        //                 listItem.classList.add('selected');
        //             }
        //         },

                
        //         move(event) {
                    
        //             const parentWidth = downloadList.offsetWidth * currentZoomLevel;
        //             const pixelsPerSecond = parentWidth / originalLengthInSeconds;

        //             const deltaStartTime = event.deltaRect.left / pixelsPerSecond;
        //             const deltaEndTime = event.deltaRect.width / pixelsPerSecond;

        //             selectedTracks.forEach(track => {
        //                 let meta = track._meta;
        //                 const wavesurfer = track._wavesurfer;
        //                 const fullDuration = wavesurfer.getDuration();

        //                 if (event.edges.left) {
        //                     const deltaTrim = event.deltaRect.left / pixelsPerSecond;
        //                     const newTrimStart = meta.trimStart + deltaTrim;
                        
        //                     const maxTrimStart = meta.trimEnd - 0.1; // avoid negative duration
        //                     meta.trimStart = Math.max(0, Math.min(newTrimStart, maxTrimStart));
        //                 }
                        

        //                 if (event.edges.right) {
        //                     meta.trackDuration = Math.min(fullDuration, Math.max(0.1, meta.trackDuration + deltaEndTime));
        //                 }

                
                        
        //                 // if (event.edges.right) {
        //                 //     const deltaTrim = event.deltaRect.width / pixelsPerSecond;
        //                 //     const newTrimEnd = meta.trimEnd + deltaTrim;
        //                 //     const maxTrimEnd = meta.startPositionInSeconds + meta.trackDuration;
        //                 //     meta.trimEnd = Math.min(maxTrimEnd, Math.max(meta.trimStart + 0.1, newTrimEnd));
        //                 // }
        //                 updateTrackPosition(track);


        //             });
        //         }
        //     }
        // });

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
                    const pixelsPerSecond = getPixelsPerSecond();

                    const deltaX = event.dx;
        
                    selectedTracks.forEach(track => {
                        let meta = track._meta;
                        const currentLeft = meta.startPositionInSeconds * pixelsPerSecond;
                        let newLeft = currentLeft + deltaX;
                        let newStartTime = newLeft / pixelsPerSecond;

                        // 🚫 Prevent dragging past 0 **only if** trimStart is already 0
                        if (meta.trimStart === 0 && newStartTime < 0) {
                            newStartTime = 0;
                            newLeft = 0;
                        }

                        // ✅ Apply new position
                        meta.startPositionInSeconds = newStartTime;
                        updateTrackPosition(listItem);
                        // const wasMuted = track.classList.contains('muted');
                        // track.classList.toggle('muted', wasMuted);



                    });
                }
            }
        });

        function applyTrimInteraction(selector, type) {
            interact(selector).draggable({
                listeners: {
                    start(event) {
                        const track = event.target.closest('li');
                        if (!selectedTracks.includes(track)) {
                            selectedTracks.forEach(el => el.classList.remove('selected'));
                            selectedTracks = [track];
                            track.classList.add('selected');
                        }
        
                        const startLine = track.querySelector('.start-line');
                        const endLine = track.querySelector('.end-line');
                        if (startLine) startLine.style.width = WAVEFORM_SELECTED_WIDTH;
                        if (endLine) endLine.style.width = WAVEFORM_SELECTED_WIDTH;
        
                        event.target.style.cursor = 'ew-resize';
                    },
                    move(event) {
                        const track = event.target.closest('li');
                        const meta = track._meta;
                        
                        const containerWidth = document.getElementById('download-links').offsetWidth * currentZoomLevel;
                        const pixelsPerSecond = getPixelsPerSecond();

        
                        const delta = event.dx / pixelsPerSecond;
        
                        if (type === 'start') {
                            meta.trimStart = Math.max(0, Math.min(meta.trimEnd - 0.1, meta.trimStart + delta));
                        } else {
                            const newTrimEnd = Math.max(meta.trimStart + 0.1, meta.trimEnd + delta);
                            meta.trimEnd = Math.min(meta.trackDuration, newTrimEnd);

                        }
        
                        document.querySelectorAll('.waveform-mask-left').forEach(el => {
                            el.style.background = WAVEFORM_ACTIVE_COLOR;
                            el.style.opacity = WAVEFORM_SELECTED_OPACITY;
                        });
                        document.querySelectorAll('.waveform-mask-right').forEach(el => {
                            el.style.background = WAVEFORM_ACTIVE_COLOR;
                            el.style.opacity = WAVEFORM_SELECTED_OPACITY;
                        });
        
                        updateTrackPosition(track);
                    },
                    end(event) {
                        event.target.style.cursor = 'ew-resize';
        
                        document.querySelectorAll('.waveform-mask-left').forEach(el => {
                            el.style.background = WAVEFORM_ACTIVE_COLOR;
                            el.style.opacity = WAVEFORM_UNSELECTED_OPACITY;
                        });
                        document.querySelectorAll('.waveform-mask-right').forEach(el => {
                            el.style.background = WAVEFORM_ACTIVE_COLOR;
                            el.style.opacity = WAVEFORM_UNSELECTED_OPACITY;
                        });
        
                        const track = event.target.closest('li');
                        const startLine = track.querySelector('.start-line');
                        const endLine = track.querySelector('.end-line');
                        if (startLine) startLine.style.width = WAVEFORM_UNSELECTED_WIDTH;
                        if (endLine) endLine.style.width = WAVEFORM_UNSELECTED_WIDTH;
                    }
                },
                cursorChecker: () => 'ew-resize'
            });
        }
        
        applyTrimInteraction('.start-line', 'start');
        applyTrimInteraction('.end-line', 'end');
        
          
          

        
        
        
        
        
        
        


        function syncer(){

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

        }


        if (typeof window._regenInsertIndex === 'number') {
            // Select the new track so syncer works
            selectedTracks = [listItem];
            listItem.classList.add('selected');
            listItem.querySelector('.waveform-container')?.classList.add('selected');
        
            syncer();
        }



        

        syncbutton.addEventListener("click", () => {
            syncer()

        });





        // Store original waveform color
        waveformContainer.dataset.originalColor = pastelColor;



        // Inside the filePaths.forEach((filePath, index) => { loop
        // Fixed solo button event handler


        



soloButton.addEventListener('click', (e) => {
    e.stopPropagation();
    const isSoloing = !soloButton.classList.contains('soloed');
    const downloadList = listItem.closest('[id^="download-links"]');
    
    // Find the correct tracklist container for solo buttons
    let tracklistContainer;
    if (downloadList.id === 'download-links') {
        tracklistContainer = document.querySelector('#muss .tracklist');
    } else if (downloadList.id === 'download-links2') {
        tracklistContainer = document.querySelector('#sfxs .tracklist');
    } else if (downloadList.id === 'download-links3') {
        tracklistContainer = document.querySelector('#vos .tracklist');
    }

    // Clear other solo buttons in this track group
    tracklistContainer.querySelectorAll('.soloButton').forEach(btn => {
        btn.classList.remove('soloed');
        btn.style.backgroundColor = '';
    });

    if (isSoloing) {
        // Solo this track
        soloButton.classList.add('soloed');
        soloButton.style.backgroundColor = 'rgb(125 155 221 / 94%)';

        // Mute all other tracks in this group
        downloadList.querySelectorAll('li').forEach(track => {
            const isCurrent = track === listItem;
            track.classList.toggle('muted', !isCurrent);
            track._wavesurfer.setVolume(isCurrent ? 1 : 0); // Actual audio mute
            track.querySelector('.waveform-container').style.opacity = isCurrent ? 1 : 0.5;
        });
    } else {
        // Unsolo all tracks
        downloadList.querySelectorAll('li').forEach(track => {
            track.classList.remove('muted');
            track._wavesurfer.setVolume(1);
            track.querySelector('.waveform-container').style.opacity = 1;
        });
    }
});


// Fixed mute button handler with proper visual feedback
muteButton.addEventListener('click', (e) => {
    e.stopPropagation();
    const downloadList = listItem.closest('[id^="download-links"]');
    
    // Prevent muting if any track is soloed
    if (downloadList.querySelector('.soloed')) {
        alert("Can't mute while solo is active");
        return;
    }

    const isMuted = !listItem.classList.contains('muted');

    if(isMuted){
    muteButton.style.backgroundColor = 'rgb(62 46 100)'
    }
    else{
    muteButton.style.backgroundColor = 'black'

    }



    listItem.classList.toggle('muted', isMuted);
    listItem._wavesurfer.setVolume(isMuted ? 0 : 1);

    
    // Update visual state through CSS classes only
    const waveform = listItem.querySelector('.waveform-container');
    waveform.style.background = isMuted ? '#666' : waveform.dataset.originalColor;
});

// Enhanced CSS for consistent visual states
const style = document.createElement('style');
style.textContent = `
    li.muted .waveform-container {
        opacity: 0.5 !important;
        filter: grayscale(80%) !important;
        background: #666 !important;
    }
    .soloed {
        background: rgb(125 155 221 / 94%) !important;
        color: black !important;
    }
    li:not(.muted) .waveform-container {
        opacity: 1 !important;
        filter: none !important;
    }
`;
document.head.appendChild(style);



if (typeof window._regenInsertIndex === 'number' && selectedTracks.length === 1) {
    syncer();
}
        
        
   
 });

    setTimeout(() => {
        document.querySelectorAll('#download-links li, #download-links2 li, #download-links3 li').forEach(updateTrackPosition);
    }, 0);

    // document.querySelectorAll('#download-links li, #download-links2 li, #download-links3 li').forEach(updateTrackPosition);



    ['download-links', 'download-links2', 'download-links3'].forEach(id => {
    const container = document.getElementById(id);
    if (container && !container.classList.contains('expanded')) {
        adjustMasksOnCollapse(container);
    }
    if (container) {
        container.classList.remove('expanded');
        container.style.height = '0px';

        container.querySelectorAll('li').forEach(li => {
            li.style.top = '0px';

            // 🫥 Hide masks to prevent overlap
            const leftMask = li.querySelector('.waveform-mask-left');
            const rightMask = li.querySelector('.waveform-mask-right');
            if (leftMask) leftMask.style.display = 'none';
            if (rightMask) rightMask.style.display = 'none';
        });
    }
});



    
    

    // Inside the filePaths.forEach((filePath, index) => { loop



    newbatch = false


    

    const containerWidth = document.getElementById('download-links').offsetWidth;
    // const containerWidth = 1000; // Fixed base width

    renderTimeline(originalLengthInSeconds, currentZoomLevel, containerWidth);
}

if (window.location.pathname == '/doseedo1.html' ) {

function applyZoom() {
    currentZoomLevel = parseFloat(zoomSlider.value);
    const containerWidth = document.getElementById('download-links').offsetWidth;

    renderTimeline(originalLengthInSeconds, currentZoomLevel, containerWidth);

    document.querySelectorAll('#download-links li, #download-links2 li, #download-links3 li')
        .forEach(updateTrackPosition);

    resizeCanvas();
    draw();
}


document.querySelector('.downloads').addEventListener('scroll', (e) => {
  const scrollLeft = e.target.scrollLeft;
  document.getElementById('automation-window').scrollLeft = scrollLeft;
});









function saveSession() {
    const projectName = localStorage.getItem('activeProject') || 'autosave';
    const sessionState = {
        originalLengthInSeconds,
        sceneChanges: globalSceneChanges,
        previewSrcKey: currentVideoId,
        tracks: [],
        cues: allCues.map(cue => ({
            time: cue.dataset.time,
            label: cue.dataset.label,
            type: cue.dataset.type
        })),
        historyStack,
        redoStack
    };
    console.log("💾 Saving previewSrcKey:", currentVideoId);

    document.querySelectorAll('[id^="download-links"] li').forEach(li => {
        const meta = li._meta;
        const waveform = li.querySelector('.waveform-container');

        sessionState.tracks.push({
            trackId: li.dataset.uuid,
            containerId: li.closest('[id^="download-links"]').id,
            fileUrl: meta.source,
            startTime: meta.startPositionInSeconds,
            duration: meta.trackDuration,
            fadeOutDuration: meta.fadeOutDuration,
            trimStart: meta.trimStart,
            trimEnd: meta.trimEnd,
            visual: {
                width: waveform?.style.width,
                startLineLeft: waveform?.querySelector('.start-line')?.style.left,
                endLineLeft: waveform?.querySelector('.end-line')?.style.left,
                leftMaskWidth: waveform?.querySelector('.waveform-mask-left')?.style.width,
                rightMaskWidth: waveform?.querySelector('.waveform-mask-right')?.style.width
            }
        });
    });

    localStorage.setItem(`session-${projectName}`, JSON.stringify(sessionState));
    console.log(`💾 Saved session: ${projectName}`);
}
















let downloadList = document.getElementById('download-links');





document.querySelector('.file-content a:nth-child(3)').addEventListener('click', (e) => {
    e.preventDefault();
    saveSession();
});






document.addEventListener('DOMContentLoaded', () => {

    setupGlobalControls(); // <--- Add this
    setupTrackDropZone('muss', 'download-links');    // music section
    setupTrackDropZone('sfxs', 'download-links2');   // sfx section
    setupTrackDropZone('vos', 'download-links3');    // vo section

    // loadSession();


        // this.style.animation = 'none';
        // this.offsetHeight;
        // // console.log('settings clicked')
        //         // audioDiv.classList.toggle("visible");
       
    
        // this.style.animation = 'spin 0.5s linear';
                
    
    
    
        var settingsForm = document.getElementById('duration-input');
        var promptSelect = document.getElementById('prompt-select');
        var melodyInput = document.getElementById('melodyInput');
        // var dropdown = document.querySelector('.dropdown');
        var pheading = document.getElementById('pheading');
        var dheading = document.getElementById('dheading');
        var mheading = document.getElementById('mheading');
        // var mheading = document.getElementById('ddheading');
        var audioDiv = document.getElementById('audiodiv')
        var record = document.querySelector(".record")
        var div = document.querySelector(".input-group1")
        var div1 = document.querySelector(".input-group2")
        var div2 = document.querySelector(".input-group3")

         // dropdown.style.display = 'grid';
         settingsForm.style.display = 'grid';
         promptSelect.style.display = 'grid';
         melodyInput.style.display = 'grid';
         pheading.style.display = 'grid';
         dheading.style.display = 'grid';
         mheading.style.display = 'grid';
         record.style.display = 'inline-block';
         div.style.display = 'inline';
         div1.style.display = 'inline';
         div2.style.display = 'inline';
 
 
 
         $(document).ready(function() {
             $('#prompt-select').select2();
             $('#prompt-select').on('select2:select select2:unselect', handleSelectionChange);
 
         });
         $(document).ready(function() {
             $('#prompt-select2').select2();
             $('#prompt-select2').on('select2:select select2:unselect', handleSelectionChange);
         });
    
        // audioDiv.classList.toggle("visible");
        
    
    
    
            
    
    


        document.getElementById('settings').addEventListener('click', function() {
            this.style.animation = 'none';
            this.offsetHeight;
            // console.log('settings clicked')
            audioDiv.classList.toggle("visible");
           
        
            this.style.animation = 'spin 0.5s linear';
            
            if (settingsForm.style.display === 'none' || settingsForm.style.display === '') {
                // dropdown.style.display = 'grid';
                settingsForm.style.display = 'grid';
                promptSelect.style.display = 'grid';
                melodyInput.style.display = 'grid';
                pheading.style.display = 'grid';
                dheading.style.display = 'grid';
                mheading.style.display = 'grid';
                record.style.display = 'inline-block';
                div.style.display = 'inline';
                div1.style.display = 'inline';
                div2.style.display = 'inline';
        
        
        
                $(document).ready(function() {
                    $('#prompt-select').select2();
                    $('#prompt-select').on('select2:select select2:unselect', handleSelectionChange);
        
                });
                $(document).ready(function() {
                    $('#prompt-select2').select2();
                    $('#prompt-select2').on('select2:select select2:unselect', handleSelectionChange);
                });
        
                if (isPro == false){
        
                // audioDiv.style.opacity = '30%'
                // audioDiv.style.pointerEvents = 'none'
        
        
                
                //const unlock = document.getElementById('probro')
                //unlock.style.display = 'block'
           
                }
            } else {
        
                setTimeout(() => {
                    
             
                settingsForm.style.display = 'none';
                promptSelect.style.display = 'none';
                melodyInput.style.display = 'none';
                pheading.style.display = 'none';
                dheading.style.display = 'none';
                mheading.style.display = 'none';
                record.style.display = 'none';
                div.style.display = 'none';
                div1.style.display = 'none';
                div2.style.display = 'none';
                // dropdown.style.display = 'none';
        
        
                $(document).ready(function() {
                    $('#prompt-select').select2('destroy');
        
                });
                $(document).ready(function() {
                    $('#prompt-select2').select2('destroy');
        
                });
            });
        
        
        
            }
                    

        })


        
 


});

document.getElementById('new-project').addEventListener('click', () => {
    const defaultName = 'Untitled Project';
    const projects = JSON.parse(localStorage.getItem('projects')) || [];

    let newName = prompt("Enter new project name:", defaultName);
    if (!newName) return;

    if (projects.includes(newName)) {
        alert("Project already exists.");
        return;
    }

    projects.push(newName);
    localStorage.setItem('projects', JSON.stringify(projects));
    localStorage.setItem('activeProject', newName);
    localStorage.setItem(`session-${newName}`, JSON.stringify({
        videoSrc: '',
        tracks: [],
        sceneChanges: []
    }));

    window.location.reload(); // clear and start fresh
});

document.getElementById('open-project').addEventListener('click', (e) => {
    e.preventDefault();
    window.location.href = 'dashboard.html';
  });

document.getElementById('save-project').addEventListener('click', () => {
    saveSession(); // assumes your existing `saveSession()` works
    alert("Project saved.");
});

document.getElementById('export-project').addEventListener('click', (e) => {
    e.preventDefault();
    exportProject();
  });

  

  function updateTrackPosition(track) {
    if (!track || !track._meta) return;

    const meta = track._meta;
    const before = { start: meta.startPositionInSeconds, trimStart: meta.trimStart, trimEnd: meta.trimEnd };
    const container = track.querySelector('.waveform-container');
    const containerWidth = document.getElementById('download-links').offsetWidth * currentZoomLevel;
    const pixelsPerSecond = getPixelsPerSecond();


    
    const minStartPx = 0; // Timeline absolute left
    const trimStartPx = meta.trimStart * pixelsPerSecond;
const trimEndPx = meta.trimEnd * pixelsPerSecond;
const trackStartPx = meta.startPositionInSeconds * pixelsPerSecond;
const visibleDuration = meta.trimEnd - meta.trimStart;
const visibleWidthPx = Math.min((meta.trimEnd - meta.trimStart), meta.trackDuration) * pixelsPerSecond;
const trackWidthPx = meta.trackDuration * pixelsPerSecond;

const cropLeftPx = trimStartPx;
const cropRightPx = (meta.trackDuration - meta.trimEnd) * pixelsPerSecond;

// ✅ Lock waveform position to scene start
container.style.left = `${trackStartPx}px`;
container.style.width = `${trackWidthPx}px`; 

// ✅ Show trimmed region only with start/end lines
if (!track._startLine) {
    track._startLine = document.createElement('div');
    track._startLine.className = 'start-line';
    container.appendChild(track._startLine);
}
// track._startLine.style.left = `${trimStartPx}px`;

if (!track._startHandle) {
    track._startHandle = document.createElement('div');
    track._startHandle.className = 'start-handle';
    container.appendChild(track._startHandle);
}
track._startHandle.style.left = `${trimStartPx - 5}px`;

if (!track._endLine) {
    track._endLine = document.createElement('div');
    track._endLine.className = 'end-line';
    container.appendChild(track._endLine);
}
if (!container.contains(track._endLine)) {
    container.appendChild(track._endLine);
}
track._endLine.style.left = `${trimEndPx}px`;

if (!track._endHandle) {
    track._endHandle = document.createElement('div');
    track._endHandle.className = 'end-handle';
    container.appendChild(track._endHandle);
}
track._endHandle.style.left = `${trimEndPx - 5}px`;


    // Ensure and position start-line
    if (!track._startLine) {
        track._startLine = document.createElement('div');
        track._startLine.className = 'start-line';
        container.appendChild(track._startLine);
    }
    if (track._startLine) {
        // track._startLine.style.left = `${trimStartPx}px`;
    }


    // Ensure and position start-handle (visual knob)
    if (!track._startHandle) {
        track._startHandle = document.createElement('div');
        track._startHandle.className = 'start-handle';
        container.appendChild(track._startHandle);
    }
    if (track._startHandle) {
        track._startHandle.style.left = `${trimStartPx - 5}px`;
    }
    // Ensure and position end-line
    if (!track._endLine) {
        track._endLine = document.createElement('div');
        track._endLine.className = 'end-line';
        container.appendChild(track._endLine);
    }
    track._endLine.style.left = `${trimEndPx}px`;

    // Ensure and position end-handle (optional visual knob, if needed)
    if (!track._endHandle) {
        track._endHandle = document.createElement('div');
        track._endHandle.className = 'end-handle';
        container.appendChild(track._endHandle);
    }
    track._endHandle.style.left = `${trimEndPx - 5}px`;

    // Update crop masks
    const leftMask = track._leftMask || container.querySelector('.waveform-mask-left');
    if (leftMask) leftMask.style.width = `${cropLeftPx}px`;

    const rightMask = track._rightMask || container.querySelector('.waveform-mask-right');
    if (rightMask) {
        rightMask.style.left = `${trackWidthPx - cropRightPx}px`;
        rightMask.style.width = `${cropRightPx}px`;
    }
    const clampedTrimStartPx = Math.max(trimStartPx, minStartPx);
    if (track._startLine) track._startLine.style.left = `${clampedTrimStartPx}px`;
    if (track._startHandle) track._startHandle.style.left = `${clampedTrimStartPx - 5}px`;

    // Update WaveSurfer settings
    const wavesurfer = track._wavesurfer;
    if (wavesurfer && wavesurfer.getDuration() > 0) {
        wavesurfer.setOptions({
            minPxPerSec: pixelsPerSecond,
            autoCenter: false,
            interact: false,
            hideScrollbar: true
        });
    }

        if (!isUndoing) {
    saveHistory({
        type: 'move',
        trackId: track.dataset.uuid,
        containerId: container.id,
        before,
        after: {
        start: meta.startPositionInSeconds,
        trimStart: meta.trimStart,
        trimEnd: meta.trimEnd
        }
    });
    }


}




window.addEventListener('resize', () => {
    // const downloadList = document.getElementById('download-links');
    const containerWidth = downloadList.offsetWidth * currentZoomLevel;
    const pixelsPerSecond = getPixelsPerSecond();


    // Optional: resize timeline
    renderTimeline(originalLengthInSeconds, currentZoomLevel, containerWidth);

    // Re-render all track positions
    document.querySelectorAll('#download-links li, #download-links2 li, #download-links3 li').forEach(track => {
        if (!track._meta) return;






        const meta = track._meta;
        const waveform = track.querySelector('.waveform-container');



        const visibleDuration = meta.trimEnd - meta.trimStart;
        const translateX = meta.startPositionInSeconds * pixelsPerSecond;
        const pixelWidth = visibleDuration * pixelsPerSecond;

        waveform.style.left = `${translateX}px`;
        waveform.style.width = `${pixelWidth}px`;

        updateTrackPosition(track);

        // console.log(`(Resized) Track: Start = ${meta.startPositionInSeconds.toFixed(2)}s | Duration = ${meta.trackDuration.toFixed(2)}s`);
    });

    // Optionally rerender timeline ticks
    renderTimeline(originalLengthInSeconds, currentZoomLevel, containerWidth);
});


}



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
        createDownloadLinks(audioFiles, sceneChanges, targetDownloadListId, isRegen = false);
        
        setTimeout(() => {
            const header = dropZone.closest('.trackselect').querySelector('p');
            const downloadList = document.getElementById(targetDownloadListId);
            const thisTracklist = header.nextElementSibling;
            const caretIcon = header.querySelector('.caret-icon');
            const wasAlreadyExpanded = thisTracklist.classList.contains('expanded');
            
            // Only collapse other tracklists if this one wasn't already expanded
            if (!wasAlreadyExpanded) {
                document.querySelectorAll('.tracklist.expanded').forEach(tracklist => {
                    const otherHeader = tracklist.previousElementSibling;
                    const otherCaret = otherHeader.querySelector('.caret-icon');
                    const otherDownloadList = document.getElementById(otherHeader.dataset.target);
                    
                    tracklist.classList.remove('expanded');
                    otherCaret?.classList.remove('rotated');

                    if (otherDownloadList) {
                        otherDownloadList.classList.remove('expanded');
                        otherDownloadList.style.height = '0';
                        otherDownloadList.querySelectorAll('li').forEach(li => {
                            li.style.top = '0';
                        });
                    }
                });
            }

            // Ensure this tracklist is expanded
            thisTracklist.classList.add('expanded');
            caretIcon?.classList.add('rotated');
            downloadList.classList.add('expanded');
            currentTargetContainerId = targetDownloadListId;
            
            // Calculate final height
            const trackCount = downloadList.querySelectorAll('li').length;
            const calculatedHeight = Math.max(100, trackCount * 90);
            
            // Only animate height if this wasn't already expanded
            if (!wasAlreadyExpanded) {
                // Trigger reflow before setting height for smooth transition
                void downloadList.offsetHeight;
                downloadList.style.height = `${calculatedHeight}px`;
            } else {
                // Just set the height immediately if already expanded
                downloadList.style.height = `${calculatedHeight}px`;
            }
            
            // Position tracks (immediately if already expanded, after delay if not)
            const positionTracks = () => {
                downloadList.querySelectorAll('li').forEach((li, i) => {
                    li.style.top = `${i * 60}px`;
                });
            };
            
            if (wasAlreadyExpanded) {
                positionTracks();
            } else {
                setTimeout(positionTracks, 300); // Match CSS transition duration
            }
            
        }, 50);
        
        window.trackAppendMode = false;
    });
}


let cursor = document.getElementById('timeline-cursor');
let cursor2 = document.getElementById('timeline-cursor2');
  




















function getPixelsPerSecond() {
    const containerWidth = document.getElementById('download-links').offsetWidth;
    const zoomedWidth = (containerWidth - 100) * currentZoomLevel;
    return zoomedWidth / originalLengthInSeconds;
}






function renderTimeline(videoDuration, zoomLevel, containerWidth) {
const timeline = document.getElementById('timeline-bar');




// Preserve existing cursor elements
const existingCursor = document.getElementById('timeline-cursor');
const existingCursor2 = document.getElementById('timeline-cursor2');

// Clear everything else
timeline.innerHTML = '';

// Re-add cursors (or recreate if missing)
const cursor = existingCursor || document.createElement('div');
cursor.id = 'timeline-cursor';
cursor.style.position = 'absolute';
cursor.style.top = '30px';
cursor.style.width = '1px';
cursor.style.height = '10000%';
cursor.style.background = '#fff';
cursor.style.zIndex = '10';
cursor.style.pointerEvents = 'none';

const cursor2 = existingCursor2 || document.createElement('div');
cursor2.id = 'timeline-cursor2';
cursor2.style.position = 'absolute';
cursor2.style.top = '0';
cursor2.style.left = '0';
cursor2.style.width = '1px';
cursor2.style.height = '0';
cursor2.style.borderLeft = '10px solid transparent';
cursor2.style.borderRight = '10px solid transparent';
cursor2.style.borderTop = '15px solid white';
cursor2.style.zIndex = '10';
cursor2.style.pointerEvents = 'none';
cursor2.style.transform = 'translateX(-50%)';

// Append back to timeline
timeline.appendChild(cursor);
timeline.appendChild(cursor2);

    const zoomedWidth = (containerWidth-100) * zoomLevel;
    timeline.style.width = `${zoomedWidth}px`;

    // const pixelsPerSecond = zoomedWidth / videoDuration;
    const pixelsPerSecond = getPixelsPerSecond();

    // 🔁 Determine ideal tick interval based on zoom level
    let secondInterval = 1;
    if (pixelsPerSecond < 2) secondInterval = 20;
    if (pixelsPerSecond < 5) secondInterval = 10;
    else if (pixelsPerSecond < 10) secondInterval = 5;
    else if (pixelsPerSecond < 20) secondInterval = 2;
    // if pixelsPerSecond > 20, keep 1s ticks

    // Optional: Sub-second minor ticks (visible only when zoomed in close)
    if (showMillisecondTicks && pixelsPerSecond > 60) {
        for (let t = 0; t <= videoDuration; t += 0.1) {
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

    // 🔶 Render major ticks & labels
    for (let t = 0; t <= videoDuration; t += secondInterval) {
        const tick = document.createElement('div');
        tick.className = 'tick major';
        tick.style.position = 'absolute';
        tick.style.left = `${t * pixelsPerSecond}px`;
        tick.style.height = '100%';
        tick.style.width = '1px';
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

    // 🔁 Reposition any cues
    allCues.forEach(cue => {
        cue.style.left = `${cue.dataset.time * pixelsPerSecond}px`;
        timeline.appendChild(cue);
    });

        // ✅ Add this at the end
    renderSceneRangeBars(globalSceneChanges);

}



function updateRegenButtonOpacity() {
    const hasSelection = document.querySelectorAll('.waveform-container.selected').length > 0;
    const regenBtn = document.getElementById('regen');
    if (regenBtn) {
        regenBtn.style.opacity = hasSelection ? '1' : '0.5';
    }
}


function renderCueArrow(time, label = 'Scene Change', type = 'scene', pixelsPerSecond = 1) {
  const arrow = document.createElement('div');
  arrow.className = 'scene-arrow';
  arrow.style.position = 'absolute';
arrow.style.display = 'none';
  arrow.style.left = `${time * pixelsPerSecond}px`;
  
  arrow.dataset.time = time;
  arrow.dataset.label = label;
  arrow.dataset.type = type;

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




if (window.location.pathname == '/doseedo1.html' ) {

// Only clear hasPlayed flags and pause all on seek
// When user seeks: jump to right position and pause or play if needed
videoPreviewsrc.addEventListener('play', () => {
    const ctx = initAudioContext();
    ctx.resume().then(() => {
        // Cleanup first
        document.querySelectorAll('#download-links li, #download-links2 li, #download-links3 li').forEach(track => {
            if (track._fx?.source) {
                track._fx.source.stop();
                track._fx.source.disconnect();
                track._fx = null;
            }
        });
        syncWaveSurfersToVideo();
    });
});


videoPreviewsrc.addEventListener('seeked', () => {
    document.querySelectorAll('#download-links li, #download-links2 li, #download-links3 li').forEach(track => {
        if (track._fx?.source) {
            track._fx.source.stop();
            track._fx.source.disconnect();
            track._fx = null;
        }
    });
});


videoPreviewsrc.addEventListener('timeupdate', syncWaveSurfersToVideo);
}

// Revised syncWaveSurfersToVideo function
function syncWaveSurfersToVideo() {
    const currentTime = videoPreviewsrc.currentTime;
    const ctx = initAudioContext();

    if (ctx.state === 'suspended') {
        ctx.resume().then(() => {
            console.log('Audio context resumed');
        });
    }

    document.querySelectorAll('#download-links li, #download-links2 li, #download-links3 li').forEach((track) => {
        const wavesurfer = track._wavesurfer;
        if (!track._meta || !track._decodedBuffer) return;

        const meta = track._meta;
        const trimOffsetStart = meta.startPositionInSeconds + meta.trimStart;


        const fadeEndTarget = meta.fadeToEndLine
            ? meta.trimEnd
            : meta.trackDuration + meta.fadeOutDuration;

        const trimOffsetEnd = meta.startPositionInSeconds + fadeEndTarget;



        const isInWindow = currentTime >= trimOffsetStart && currentTime <= trimOffsetEnd;
        const wasInWindow = track._fx?.isPlayingInWindow || false;

        if (wavesurfer) {
            wavesurfer.pause();
            wavesurfer.setVolume(0);
        }

        if (isInWindow && !wasInWindow) {
            try {
                const source = ctx.createBufferSource();
                source.buffer = track._decodedBuffer;

                const dryGain = ctx.createGain();
                const wetGain = ctx.createGain();
                const masterGain = ctx.createGain();
                const convolver = ctx.createConvolver();
                convolver.buffer = reverbIRBuffer;

                const amt = typeof meta.reverbAmount === 'number' ? meta.reverbAmount : 0.2;
                dryGain.gain.value = 1 - amt;
                wetGain.gain.value = amt;

                source.connect(dryGain).connect(masterGain);
                source.connect(convolver).connect(wetGain).connect(masterGain);
                
                


// ⬇️ Add this BEFORE connecting masterGain
                const isMuted = track.classList.contains('muted');
                const parentContainer = track.closest('[id^="download-links"]');
                const soloedTrack = parentContainer?.querySelector('.soloed');
                const isSoloed = soloedTrack && soloedTrack.closest('li') === track;

                const shouldMute = isMuted || (soloedTrack && !isSoloed);
                if (shouldMute) {
                    console.log('🔇 Skipping playback due to mute/solo state');
                    return;
                }

                // ✅ Only connect to output if allowed
                masterGain.connect(ctx.destination);
     

                            
                const playbackTrimEnd = meta.fadeToEndLine
                    ? meta.trimEnd
                    : Math.min(meta.trackDuration + meta.fadeOutDuration, track._decodedBuffer.duration);

                const trackEndInVideo = meta.startPositionInSeconds + meta.trimEnd + (
                    meta.fadeToEndLine ? 0 : meta.fadeOutDuration
                );


                    const secondsUntilTrackEnd = trackEndInVideo - currentTime;


                // Playback timing
                const offset = Math.max(0, currentTime - trimOffsetStart);
                const startTime = ctx.currentTime;
                const duration = Math.min(
                    playbackTrimEnd - meta.trimStart - offset,
                    track._decodedBuffer.duration - meta.trimStart
                );

                // ⏳ Fade-in (default 0.3s unless overridden)
                

                let fadeInDuration = meta.fadeInDuration || 0.3;
                let fadeOutDuration = Math.min(secondsUntilTrackEnd, meta.fadeOutDuration || 1.0);
                let fadeInEndTime = startTime + fadeInDuration;
                let fadeOutStartTime = ctx.currentTime + (secondsUntilTrackEnd - fadeOutDuration);
                let fadeOutEndTime = fadeOutStartTime + fadeOutDuration;


                                // 🚫 Prevent overlap: Ensure fade-in ends before fade-out starts
                if (fadeOutStartTime < fadeInEndTime) {
                    console.warn('⚠️ Fade in and out overlap — adjusting fade durations');
                    const totalAvailable = fadeOutEndTime - startTime;

                    const half = totalAvailable / 2;
                    fadeInDuration = half;
                    fadeOutDuration = half;

                    // Recalculate times
                    fadeInEndTime = startTime + fadeInDuration;
                    fadeOutStartTime = fadeInEndTime;
                    fadeOutEndTime = fadeOutStartTime + fadeOutDuration;
                }



                console.log(`🎧 fadeIn = ${fadeInDuration}s, fadeOut = ${fadeOutDuration}s`);

                // 🎚 Schedule fades
                masterGain.gain.cancelScheduledValues(startTime);


                // 🧠 Dynamic gain for mid-fade-in seeks
                let initialGain = 1.0;

                if (fadeInDuration > 0.01) {
                    const fadeInProgress = offset / fadeInDuration;
                    if (fadeInProgress < 1.0) {
                        initialGain = Math.max(0.001, fadeInProgress);  // Avoid zero gain
                    }
                }

                // 📌 Apply safe starting gain
                masterGain.gain.setValueAtTime(initialGain, startTime);

                // 🌀 Continue ramping if still within fade-in
                if (offset < fadeInDuration) {
                    masterGain.gain.linearRampToValueAtTime(1.0, fadeInEndTime);
                }


                


                masterGain.gain.setValueAtTime(1.0, fadeOutStartTime); // hold
                masterGain.gain.exponentialRampToValueAtTime(0.001, fadeOutEndTime); // fade out

                // 🔊 Start playback
                source.start(startTime, meta.trimStart + offset, duration);

                track._fx = {
                    source,
                    dryGain,
                    wetGain,
                    masterGain,
                    convolver,
                    isPlayingInWindow: true
                };
            } catch (e) {
                console.error("Error starting track:", e);
            }
        } else if (!isInWindow && wasInWindow) {
            try { track._fx.source?.stop(); } catch (e) {}
            track._fx.source?.disconnect();
            track._fx = null;
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
                      totalDuration = originalLengthInSeconds;
                    // const containerWidth = 1000; // Fixed base width
        
                    const containerWidth = document.getElementById('download-links').offsetWidth;
                   // Calculate zoom level to fit entire video in view
                    const fitToScreenZoomLevel = containerWidth / originalLengthInSeconds;
                    currentZoomLevel = fitToScreenZoomLevel;

                    const vidx = document.getElementById('vidx');
                    vidx.style.display = 'block';
                
                    let zoomSlider = document.getElementById("zoom-slider");

                    zoomSlider.min = 1; // or 0.1 if you want more zoom-in range
                    zoomSlider.max = 10;
                    zoomSlider.step = 0.1;
                    // zoomSlider.value = fitToScreenZoomLevel;
                    zoomSlider.value = "1";
                    currentZoomLevel = 1
                
                   renderTimeline(originalLengthInSeconds, currentZoomLevel, containerWidth);
                
                    sceneChanges = [0, 2, 4];
                    // const audioFiles = ['Galaga.mp3', 'Galaga.mp3', 'Galaga.mp3'];
                    // createDownloadLinks(audioFiles, sceneChanges);
                
                    // player.play();
                };
                
                
            };
    
            reader.readAsDataURL(file);
        } else {
            videoPreviewsrc.src = '';
        }
    });
    



const videoElement = document.getElementById('player');
const timelineCursor = document.getElementById('timeline-cursor');

function updateTimelineCursor() {
  const containerWidth = document.getElementById('download-links').offsetWidth * currentZoomLevel;
  const pixelsPerSecond = getPixelsPerSecond();

  const cursorX = videoElement.currentTime * pixelsPerSecond;

  timelineCursor.style.left = `${cursorX}px`;

  requestAnimationFrame(updateTimelineCursor);
}

videoElement.addEventListener('play', () => {
  console.log('🎬 Video is playing — starting cursor animation');
  requestAnimationFrame(updateTimelineCursor);
});








    


videoPreviewsrc.addEventListener('pause', () => {
    document.querySelectorAll('#download-links li, #download-links2 li, #download-links3 li').forEach(track => {
        if (track._fx?.source) {
            track._fx.source.stop();
            track._fx = null;
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

let isUndoing = false;

function setupUndoRedo(updateFn) {
  function undo() {
    const last = historyStack.pop();
    if (!last) return;
    isUndoing = true;
    applyAction(last, 'undo');
    isUndoing = false;
    redoStack.push(last);
  }

  function redo() {
    const next = redoStack.pop();
    if (!next) return;
    isUndoing = true;
    applyAction(next, 'redo');
    isUndoing = false;
    historyStack.push(next);
  }

  function applyAction(action, direction) {
    const track = document.querySelector(`[data-uuid="${action.trackId}"]`);
    if (!track || !track._meta) return;
    const meta = track._meta;
    const state = direction === 'undo' ? action.before : action.after;

    meta.startPositionInSeconds = state.start;
    meta.trimStart = state.trimStart;
    meta.trimEnd = state.trimEnd;
    updateFn(track);
  }

  return { undo, redo };
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

    const isGoogle = !!googleProfile;
    const effectiveUsername = isGoogle ? googleProfile.getName() : username;
    const effectiveEmail = isGoogle ? googleProfile.getEmail() : email;
    const effectivePassword = isGoogle ? googleProfile.getId() : password;
    const effectivePicture = isGoogle ? googleProfile.getPicture() : 'user.png';

    formData.append('username', effectiveUsername);
    formData.append('email', effectiveEmail);
    formData.append('password', effectivePassword);

    fetch('https://doseedo.com/register/', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            document.querySelector('#signup-error-message').innerText = 'Username or email already exists.';
            document.querySelector('#signup-error-message').style.display = 'block';
            throw new Error('Invalid data.');
        }
        return response.json();
    })
    .then(data => {
        console.log('Registration success:', data);
        isAuthenticated = true;

        const subscriptionStatus = data.subscription ? "Pro+" : "Free";

        localStorage.setItem('isAuth', 'true');
        localStorage.setItem('username', effectiveUsername);
        localStorage.setItem('email', effectiveEmail);
        localStorage.setItem('userpic', effectivePicture);
        localStorage.setItem('ispro', subscriptionStatus);

        document.getElementById('user-subscription-status').textContent = subscriptionStatus;
        document.querySelector('#signup-error-message').innerText = 'Successfully registered';
        document.querySelector('#signup-error-message').style.color = 'white';

        // Track sign-up in Google Analytics
        if (window.gtag) {
            window.gtag('event', 'sign_up', { method: isGoogle ? 'google' : 'email' });
        }

        window.window.location.href = "/dashboard";
    })
    .catch(error => {
        console.error('Registration error:', error);
        document.querySelector('#signup-error-message').innerText = error;
        document.querySelector('#signup-error-message').style.display = 'block';
    });
}



function loginUser(googleProfile) {
    let formData = new FormData();
    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;

    const isGoogle = !!googleProfile;
    const effectiveEmail = isGoogle ? googleProfile.getEmail() : username;
    const effectivePassword = isGoogle ? googleProfile.getId() : password;
    const effectivePicture = isGoogle ? googleProfile.getPicture() : 'user.png';

    formData.append('username', effectiveEmail);
    formData.append('password', effectivePassword);

    fetch('https://doseedo.com/token', {
        credentials: 'include',  // Required to accept cookies
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            document.querySelector('#signin-error-message').innerText = 'Incorrect username or password';
            document.querySelector('#signin-error-message').style.display = 'block';
            throw new Error('Incorrect username or password');
        }
        return response.json();
    })
    .then(data => {
        console.log('Login success:', data);
        isAuthenticated = true;

        const subscriptionStatus = data.subscription ? "Pro+" : "Free";

        localStorage.setItem('isAuth', 'true');
        localStorage.setItem('username', effectiveEmail);
        localStorage.setItem('userpic', effectivePicture);
        localStorage.setItem('ispro', subscriptionStatus);

        document.getElementById('user-username').textContent = effectiveEmail;
        document.getElementById('user-subscription-status').textContent = subscriptionStatus;

        document.getElementById('user-info').style.display = 'flex';
        document.getElementById('register-form').style.display = 'none';
        document.getElementById('login-form').style.display = 'none';
        document.querySelector('.formdiv').style.display = 'none';

        window.location.href = "/dashboard";
    })
    .catch(error => {
        if (isGoogle) {
            registerUser(googleProfile); // fallback: attempt registration
        } else {
            console.error('Login error:', error);
            document.querySelector('#signin-error-message').innerText = error;
            document.querySelector('#signin-error-message').style.display = 'block';
        }
    });
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
            window.location.href = "/dashboard";
        }, 1000); // Time in milliseconds (10000 milliseconds = 10 seconds)
        // You can add additional code here to handle successful subscription storage
    })
    .catch(error => console.error('Error storing subscription:', error));
}

function signOut() {
    
    isAuthenticated = false;
    toggleSignOutButton();
    localStorage.removeItem('isAuth');
    localStorage.removeItem('username');
    localStorage.removeItem('ispro');
    window.window.location.href = "/dashboard";
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

// Add to initialization
// let audioContext;
// let audioContextState = 'unknown';



function isTrackExportReady(track) {
  return track.dataset.ready === 'true' && 
         track._wavesurfer?.backend?.buffer &&
         track._meta?.startPositionInSeconds !== undefined;
}




// Initialize on page load
// const audioContext = monitorAudioContext();





// Optimized async function to export all tracks to a combined audio (WAV) for video
async function exporttovid() {
    const tracks = document.querySelectorAll('#download-links li, #download-links2 li, #download-links3 li');
    const audioFiles = [];
    const trackMetadata = [];

    for (const track of tracks) {
        const buffer = track._decodedBuffer;
        const meta = track._meta;

        if (!buffer || !meta) {
            console.warn('⚠️ No decoded buffer or meta for this track, skipping:', track);
            continue;
        }

        if (track.classList.contains('muted')) {
            console.log('🔇 Track is muted, skipping:', track);
            continue;
        }

        try {
            const wavBlob = bufferToWavBlob(buffer);
            const file = new File([wavBlob], `track_${Date.now()}.wav`, { type: 'audio/wav' });
            const { startPositionInSeconds, trackDuration } = meta;

            audioFiles.push(file);

            trackMetadata.push({
                start: startPositionInSeconds,
                duration: trackDuration,
                filename: file.name
            });
        } catch (err) {
            console.error('❌ Error during buffer export for track:', err);
        }
    }

    if (audioFiles.length === 0) {
        alert('❌ No active audio tracks available for export.');
        return;
    }

    console.log('✅ Prepared audioFiles and metadata:', { audioFiles, trackMetadata });

    const formData = new FormData();

    const videoInput = document.getElementById('videoFile');
    if (!videoInput.files.length) {
        alert('❌ No video selected.');
        return;
    }
    formData.append('video', videoInput.files[0]);
    audioFiles.forEach((file, index) => {
        formData.append(`file_${index}`, file);
    });
    formData.append('audioTracks', JSON.stringify(trackMetadata));

    try {
        updateStatus('Sending tracks for export...');
        loading.style.display = 'block';

        const response = await fetch('https://doseedo.com/export/', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            console.log('✅ Export started:', result);
            // alert('🎬 Export started! Task ID: ' + result.task_id);

            // ✅ Start polling now
            pollExportStatus(result.task_id, result.temp_dir);
        } else {
            console.error('❌ Export failed:', result);
            alert('❌ Export failed.');
        }
    } catch (error) {
        console.error('❌ Error sending to backend:', error);
        alert('❌ Failed to send export request.');
    } finally {
        loading.style.display = 'none';
    }
}

// NEW FUNCTION: Poll backend if export is done
async function pollExportStatus(taskId, tempDir) {
    console.log('🕓 Starting polling for export task:', taskId);

    const pollInterval = 3000; // 3 seconds

    const intervalId = setInterval(async () => {
        try {
            const response = await fetch(`https://doseedo.com/export/status/${taskId}`);

            if (!response.ok) {
                console.error('❌ Failed to get export status.');
                return;
            }

            const result = await response.json();

            console.log('🔍 Export status response:', result);

            if (result.status === 'SUCCESS') {
                clearInterval(intervalId);
                console.log('🎉 Export finished!', result);

                const downloadUrl = `https://doseedo.com/${tempDir}/final_output.mp4`;

                // alert('✅ Export finished! Download your video now.');
                window.open(downloadUrl, '_blank');
            } else if (result.status === 'FAILURE') {
                clearInterval(intervalId);
                alert('❌ Export failed.');
            } else {
                console.log('⏳ Still processing...');
            }

        } catch (error) {
            console.error('❌ Error polling export status:', error);
            clearInterval(intervalId);
        }
    }, pollInterval);
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
    
    
    // document.getElementById('showCuesBtn').style.visibility = 'visible'
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

videoFileElement2.addEventListener('mouseover', function(event) {
   
    icon2.classList.add('expanding-from-center');
    icon2.classList.remove('shrinking-to-center');
    icon2.classList.add('expanding-from-center');
    icon2.classList.remove('shrinking-to-center');
    document.querySelector('.glow-container3');
});

videoFileElement2.addEventListener('mouseout', function(event) {

    icon2.classList.remove('expanding-from-center');
    icon2.classList.add('shrinking-to-center');
});


videoFileElement3.addEventListener('mouseover', function(event) {
   
    icon3.classList.add('expanding-from-center');
    icon3.classList.remove('shrinking-to-center');
    icon3.classList.add('expanding-from-center');
    icon3.classList.remove('shrinking-to-center');
    document.querySelector('.glow-container3');
});

videoFileElement3.addEventListener('mouseout', function(event) {

    icon3.classList.remove('expanding-from-center');
    icon3.classList.add('shrinking-to-center');
});

videoFileElement4.addEventListener('mouseover', function(event) {
   
    icon4.classList.add('expanding-from-center');
    icon4.classList.remove('shrinking-to-center');
    icon4.classList.add('expanding-from-center');
    icon4.classList.remove('shrinking-to-center');
    document.querySelector('.glow-container3');
});

videoFileElement4.addEventListener('mouseout', function(event) {

    icon4.classList.remove('expanding-from-center');
    icon4.classList.add('shrinking-to-center');
});



}




function resetAnimation() {
    icon.classList.remove('expanding-from-center');
    icon.classList.add('shrinking-to-center');
}

function onSignIn(googleUser) {
    const profile = googleUser.getBasicProfile();
    loginUser(profile); // Always try login first
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


if (window.location.pathname == '/doseedo1.html' ) {


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




  function resizeTimeline() {
    const zoomLevel = parseFloat(currentZoomLevel || 1);
    
    const containerWidth = document.getElementById('download-links').offsetWidth;
    // const containerWidth = 1000; // Fixed base width
    renderTimeline(originalLengthInSeconds, currentZoomLevel, containerWidth);

}





    
        window.addEventListener('resize', resizeTimeline);
        resizeTimeline(); // Call once on load










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

}









// Call this function to initially populate the projects list

// Resizable container sync functionality
function initResizableSync() {
    const trackContainer = document.querySelector('.trackcontainer');
    const downloadsContainer = document.querySelector('.downloads');

    if (!trackContainer || !downloadsContainer) return;

    // Create ResizeObserver to watch for size changes
    const resizeObserver = new ResizeObserver(entries => {
        for (let entry of entries) {
            const newHeight = entry.contentRect.height;

            // Update CSS custom property for reactive elements
            document.documentElement.style.setProperty('--container-height', newHeight + 'px');

            // Sync the other container's height
            if (entry.target === trackContainer && downloadsContainer.style.height !== newHeight + 'px') {
                downloadsContainer.style.height = newHeight + 'px';
            } else if (entry.target === downloadsContainer && trackContainer.style.height !== newHeight + 'px') {
                trackContainer.style.height = newHeight + 'px';
            }
        }
    });

    // Start observing both containers
    resizeObserver.observe(trackContainer);
    resizeObserver.observe(downloadsContainer);

    // Also listen for manual resize events (CSS resize property)
    function handleResize(element) {
        const height = element.offsetHeight;
        document.documentElement.style.setProperty('--container-height', height + 'px');

        // Sync the other container
        if (element === trackContainer) {
            downloadsContainer.style.height = height + 'px';
        } else {
            trackContainer.style.height = height + 'px';
        }
    }

    // Set up mutation observer to detect style changes (for CSS resize)
    const mutationObserver = new MutationObserver(mutations => {
        mutations.forEach(mutation => {
            if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
                handleResize(mutation.target);
            }
        });
    });

    mutationObserver.observe(trackContainer, { attributes: true, attributeFilter: ['style'] });
    mutationObserver.observe(downloadsContainer, { attributes: true, attributeFilter: ['style'] });
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', initResizableSync);

