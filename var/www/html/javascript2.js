// JavaScript functions for ACE-Step model interface
let isAuthenticated = false
let using = ''
let isPro = false
let mus = false
let currentVideoId = ''
let videolabels = ''
let sceneChanges = null  // Store scene change timestamps from video analysis
let automationData = null  // Store volume automation data from video analysis

// Global sync lock to prevent circular video/audio play/pause calls
window.videoAudioSyncLock = false;

// Auth and UI initialization from original javascript.js
window.onload = function() {
    const isAuth = localStorage.getItem('isAuth') === 'true';
    const username = localStorage.getItem('username');
    const substatus = localStorage.getItem('ispro');
    using = localStorage.getItem('username');

    if (isAuth && username) {
        isAuthenticated = true
        if(substatus == 'Pro+'){
            isPro = true
        }

        const userUsername = document.getElementById('user-username');
        const userSubscriptionStatus = document.getElementById('user-subscription-status');
        const userInfo = document.getElementById('user-info');
        const registerForm = document.getElementById('register-form');
        const loginForm = document.getElementById('login-form');
        const signupButton = document.getElementById('signupbutton');
        const loginButton = document.getElementById('loginbutton');

        if (userUsername) userUsername.textContent = username;
        if (userSubscriptionStatus) userSubscriptionStatus.textContent = substatus;
        if (userInfo) userInfo.style.display = 'block';
        if (registerForm) registerForm.style.display = 'none';
        if (loginForm) loginForm.style.display = 'none';
        if (signupButton) signupButton.style.display = 'none';
        if (loginButton) loginButton.style.display = 'none';

        toggleSignOutButton();
    }
};

function showSignUp() {
    const registerForm = document.getElementById('register-form');
    const loginForm = document.getElementById('login-form');
    const userInfo = document.getElementById('user-info');
    const background = document.getElementById('background');

    if (registerForm) registerForm.style.display = 'block';
    if (loginForm) loginForm.style.display = 'none';
    if (userInfo) userInfo.style.display = 'none';
    if (background) background.style.opacity = '0.1';
}

function showSignIn() {
    const registerForm = document.getElementById('register-form');
    const loginForm = document.getElementById('login-form');
    const userInfo = document.getElementById('user-info');
    const background = document.getElementById('background');

    if (registerForm) registerForm.style.display = 'none';
    if (loginForm) loginForm.style.display = 'block';
    if (userInfo) userInfo.style.display = 'none';
    if (background) background.style.opacity = '0.1';
}

function exitregistration(){
    const registerForm = document.getElementById('register-form');
    const loginForm = document.getElementById('login-form');
    const background = document.getElementById('background');

    if (registerForm) registerForm.style.display = 'none';
    if (loginForm) loginForm.style.display = 'none';
    if (background) background.style.opacity = '1';
}

// Settings toggle with ACE-Step model parameters
const settingsBtn = document.getElementById('settings');
if (settingsBtn) {
    settingsBtn.addEventListener('click', function() {
        this.style.animation = 'none';
        this.offsetHeight;
        this.style.animation = 'spin 0.5s linear';

        var modelSettings = document.getElementById('model-settings');
        var modelHeading = document.getElementById('model-heading');
        var durationInput = document.getElementById('duration-input');
        var melodyInput = document.getElementById('melodyInput');
        var dheading = document.getElementById('dheading');
        var mheading = document.getElementById('mheading');
        var audioDiv = document.getElementById('audiodiv');

        if (audioDiv) audioDiv.classList.toggle("visible");

        if (modelSettings && modelSettings.style.display === 'none' || modelSettings && modelSettings.style.display === '') {
            if (modelSettings) modelSettings.style.display = 'grid';
            if (modelHeading) modelHeading.style.display = 'grid';
            if (durationInput) durationInput.style.display = 'grid';
            if (melodyInput) melodyInput.style.display = 'grid';
            if (dheading) dheading.style.display = 'grid';
            if (mheading) mheading.style.display = 'grid';

            if (isPro == false && audioDiv){
                audioDiv.style.opacity = '30%'
                audioDiv.style.pointerEvents = 'none'
            }
        } else {
            setTimeout(() => {
                if (modelSettings) modelSettings.style.display = 'none';
                if (modelHeading) modelHeading.style.display = 'none';
                if (durationInput) durationInput.style.display = 'none';
                if (melodyInput) melodyInput.style.display = 'none';
                if (dheading) dheading.style.display = 'none';
                if (mheading) mheading.style.display = 'none';
            }, 500);
            if (audioDiv) audioDiv.style.opacity = '100%'
        }
    });
}

function music(){
    console.log(mus);
    mus = true;
    console.log(mus);
}

function handleGenerateMusic() {
    // Smart workflow: upload video if present, otherwise generate directly
    const videoInput = document.getElementById('videoFile');

    if (videoInput && videoInput.files.length > 0) {
        // Video uploaded - upload it first for scene analysis
        console.log('🎬 Video detected - uploading for analysis');
        music();  // Set flag to generate after upload
        uploadVideo();
    } else {
        // No video - generate music directly (will use MIDI or reference audio)
        console.log('🎹 No video - generating music directly');
        generateMusic();
    }
}

function updateStatus(message) {
    const statusEl = document.getElementById('status');
    if (statusEl) statusEl.textContent = message;
}

function uploadVideo() {
    updateStatus('Uploading Video');
    const videoInput = document.getElementById('videoFile');
    if (!videoInput || videoInput.files.length === 0) {
        alert('Please select a video file.');
        return;
    }

    // Clear previous session data
    automationData = null;
    sceneChanges = null;
    console.log('🔄 Cleared previous automation and scene data');

    const formData = new FormData();
    formData.append('file', videoInput.files[0]);

    fetch('https://doseedo.com/uploadvideo/', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        updateStatus('Uploaded Video');
        if (data.video_id) {
            currentVideoId = data.video_id
            pollTaskStatus2(data.task_id);
        } else {
            throw new Error('Video ID not received in response');
        }
    })
    .catch(error => {
        console.error('Error during video upload:', error);
        updateStatus('Error Uploading');
    });
}

function pollTaskStatus2(taskId) {
    const statusInterval = setInterval(() => {
        fetch(`https://doseedo.com/task-status/${taskId}`)
        .then(response => response.json())
        .then(data => {
            updateStatus(data.status);
            if(data.status === 'SUCCESS') {
                clearInterval(statusInterval);
                console.log('Video analysis labels:', data.result);

                const descriptionInput = document.getElementById('description-input');
                if (descriptionInput) {
                    descriptionInput.value = data.result.labels.join(', ');
                    videolabels = data.result.labels.join(', ');
                    console.log(descriptionInput.value);
                }

                // Extract scene changes and automation data if available
                if (data.result.scene_changes) {
                    sceneChanges = data.result.scene_changes;
                    console.log('🎬 Scene changes detected:', sceneChanges);
                }
                if (data.result.automation_data) {
                    automationData = data.result.automation_data;
                    console.log('🎚️ Automation data extracted:', automationData);
                }

                if (mus) {
                    generateMusic();
                    mus = false;
                }
            } else if(data.status === 'FAILURE') {
                clearInterval(statusInterval);
                throw new Error('Error in video processing');
            }
        })
        .catch(error => {
            clearInterval(statusInterval);
            console.error('Error:', error);
            updateStatus('Error during task status check.');
        });
    }, 3000);
}

// Track generation counters for unique IDs
let musicGenerationCount = 0;

/**
 * Creates a new download-links UL for a generation
 * @param {string} type - 'music', 'sfx', or 'vo'
 * @returns {string} - The ID of the new download-links container
 */
function createNewTrackselect(type) {
    const downloadsDiv = document.querySelector('.downloads');
    if (!downloadsDiv) {
        console.error('.downloads container not found');
        return null;
    }

    // Generate unique ID for this generation
    const counter = ++musicGenerationCount;
    const newContainerId = `download-links-gen${counter}`;

    // Create new UL element for tracks
    const newUL = document.createElement('ul');
    newUL.id = newContainerId;
    newUL.className = 'download-list';

    // Apply inline styles to match existing download-links styling
    newUL.style.cssText = `
        text-align: center;
        position: relative;
        border-radius: 10px;
        color: rgb(168, 115, 221);
        display: flex;
        flex-direction: column;
        font-family: monospace;
        height: auto;
        overflow: visible;
        transition: margin-top 0.3s ease, height 0.3s ease;
        padding-right: 50px;
        width: 100%;
        margin-top: 20px;
        font-family: 'Gill Sans', 'Gill Sans MT', Calibri, 'Trebuchet MS', sans-serif;
    `;

    // Add a generation label/badge
    const genLabel = document.createElement('div');
    genLabel.style.cssText = 'color: rgba(255,255,255,0.5); font-size: 0.9em; padding: 5px 0; margin-left: 50px;';
    genLabel.textContent = `Generation #${counter}`;

    // Insert label before the UL
    downloadsDiv.appendChild(genLabel);
    downloadsDiv.appendChild(newUL);

    console.log(`✅ Created new download-links UL: ${newContainerId}`);

    return newContainerId;
}

// Main generate music function for ACE-Step model
function generateMusic() {
    const melodyInput = document.getElementById('melodyInput');
    const videoInput = document.getElementById('videoFile');
    const durationInput = document.getElementById('duration-input');

    // Now optional: conditioning file can be audio/MIDI, video, or auto-generated MIDI
    // If no file is provided, backend will automatically generate MIDI based on duration
    const hasFile = (melodyInput && melodyInput.files.length > 0) || (videoInput && videoInput.files.length > 0);

    if (!hasFile) {
        // Check if duration is set for MIDI generation
        if (!durationInput || !durationInput.value || parseFloat(durationInput.value) <= 0) {
            alert('Please either:\n1. Upload a video file, OR\n2. Upload a conditioning audio/MIDI file, OR\n3. Set a duration (auto-generate MIDI conditioning)');
            updateStatus('Waiting for input...');
            return;
        }
        console.log('🎹 No file provided - backend will auto-generate MIDI conditioning');

        // Clear automation/scene data when generating without video
        automationData = null;
        sceneChanges = null;
        console.log('🔄 No video - cleared automation and scene data');
    }

    updateStatus('Generating music with ACE-Step model...');

    const descriptionInput = document.getElementById('description-input');

    // Gather all ACE-Step model parameters with defaults if elements don't exist
    const stepsEl = document.getElementById('steps-input');
    const seedEl = document.getElementById('seed-input');
    const adapterScaleEl = document.getElementById('adapter-scale-input');
    const cfgWeightEl = document.getElementById('cfg-weight-input');
    const instrumentStrengthEl = document.getElementById('instrument-strength-input');
    const noiseLevelEl = document.getElementById('noise-level-input');
    const pianoRollGainEl = document.getElementById('piano-roll-gain-input');
    const ampGainEl = document.getElementById('amp-gain-input');
    const rframeGainEl = document.getElementById('rframe-gain-input');
    const rbendGainEl = document.getElementById('rbend-gain-input');
    const encodecGainEl = document.getElementById('encodec-gain-input');
    const pitchFidelityEl = document.getElementById('pitch-fidelity-input');
    const onsetGuidanceEl = document.getElementById('onset-guidance-input');
    const pitchSnapEl = document.getElementById('pitch-snap-input');

    const steps = stepsEl ? stepsEl.value : '50';
    const seed = seedEl ? seedEl.value : '-1';
    const adapterScale = adapterScaleEl ? adapterScaleEl.value : '1.0';
    const cfgWeight = cfgWeightEl ? cfgWeightEl.value : '3.0';
    const instrumentStrength = instrumentStrengthEl ? instrumentStrengthEl.value : '1.0';
    const noiseLevel = noiseLevelEl ? noiseLevelEl.value : '0.0';
    const pianoRollGain = pianoRollGainEl ? pianoRollGainEl.value : '1.0';
    const ampGain = ampGainEl ? ampGainEl.value : '1.0';
    const rframeGain = rframeGainEl ? rframeGainEl.value : '1.0';
    const rbendGain = rbendGainEl ? rbendGainEl.value : '1.0';
    const encodecGain = encodecGainEl ? encodecGainEl.value : '1.0';
    const pitchFidelity = pitchFidelityEl ? pitchFidelityEl.value : '0.0';
    const onsetGuidance = onsetGuidanceEl ? onsetGuidanceEl.value : '0.0';
    const pitchSnap = pitchSnapEl ? pitchSnapEl.value : '0.0';

    // Get instrument selection for MIDI rendering
    const instrumentGroupEl = document.getElementById('instrument-group');
    const instrumentSubgroupEl = document.getElementById('instrument-subgroup');
    const instrumentGroup = instrumentGroupEl ? instrumentGroupEl.value : 'strings';
    const instrumentSubgroup = instrumentSubgroupEl ? instrumentSubgroupEl.value : 'violin';

    const formData = new FormData();
    formData.append('description', descriptionInput ? descriptionInput.value : '');
    formData.append('duration', durationInput ? durationInput.value : '30');
    formData.append('steps', steps);
    formData.append('seed', seed);
    formData.append('adapter_scale', adapterScale);
    formData.append('cfg_weight', cfgWeight);
    formData.append('instrument_strength', instrumentStrength);
    formData.append('noise_level', noiseLevel);
    formData.append('piano_roll_gain', pianoRollGain);
    formData.append('amp_gain', ampGain);
    formData.append('rframe_gain', rframeGain);
    formData.append('rbend_gain', rbendGain);
    formData.append('encodec_gain', encodecGain);
    formData.append('pitch_fidelity_boost', pitchFidelity);
    formData.append('onset_guidance_boost', onsetGuidance);
    formData.append('pitch_snap_strength', pitchSnap);

    // Add instrument selection for FluidSynth MIDI rendering (when no audio uploaded)
    formData.append('instrument_group', instrumentGroup);
    formData.append('instrument_subgroup', instrumentSubgroup);

    // Add scene data if available from video analysis
    // Get actual video duration from video element
    const videoElement = document.getElementById('player');
    const actualVideoDuration = videoElement && videoElement.duration > 0 ? videoElement.duration : null;
    const userDuration = durationInput && durationInput.value ? parseFloat(durationInput.value) : null;
    const targetDuration = userDuration || actualVideoDuration;

    console.log(`📹 Video duration: ${actualVideoDuration}s, User duration: ${userDuration}s, Target: ${targetDuration}s`);

    // If we have scene changes AND a target duration, extend scene durations to full video
    if (sceneChanges && sceneChanges.length > 1 && targetDuration) {
        // Convert scene_changes to scene_durations
        const sceneDurations = [];
        for (let i = 0; i < sceneChanges.length - 1; i++) {
            sceneDurations.push(sceneChanges[i + 1] - sceneChanges[i]);
        }

        // Add final segment from last scene change to end of video/target duration
        const lastSceneTime = sceneChanges[sceneChanges.length - 1];
        const finalSegmentDuration = targetDuration - lastSceneTime;

        if (finalSegmentDuration > 0.5) {  // Only add if significant duration remains
            sceneDurations.push(finalSegmentDuration);
            console.log(`🎬 Scene durations (extended to ${targetDuration}s):`, sceneDurations);
            console.log(`   Added final segment: ${finalSegmentDuration.toFixed(2)}s`);
        } else {
            console.log(`🎬 Scene durations:`, sceneDurations);
        }

        formData.append('scene_durations', JSON.stringify(sceneDurations));
    }

    // Add automation data if available
    if (automationData) {
        formData.append('automation_data', JSON.stringify(automationData));
        console.log('🎚️ Sending automation data');
    }

    // Add conditioning file: prefer audio/MIDI input, fallback to video, or auto-generate MIDI
    console.log('🎵 File check:');
    console.log('  melodyInput:', melodyInput);
    console.log('  videoInput:', videoInput);
    console.log('  videoInput.files.length:', videoInput ? videoInput.files.length : 'N/A');
    console.log('  videoInput.files[0]:', videoInput && videoInput.files.length > 0 ? videoInput.files[0] : 'N/A');

    if (melodyInput && melodyInput.files.length > 0) {
        console.log('✅ Appending melodyInput file');
        formData.append('audio_file', melodyInput.files[0]);
    } else if (videoInput && videoInput.files.length > 0) {
        console.log('✅ Appending videoInput file');
        formData.append('audio_file', videoInput.files[0]);
    } else {
        console.log('🎹 No file - backend will auto-generate MIDI conditioning');
        console.log(`🎻 Target instrument: ${instrumentGroup} > ${instrumentSubgroup}`);
        console.log('🎼 Backend will render MIDI with FluidSynth using selected instrument');
        // No file needed - backend will generate MIDI based on duration and render with FluidSynth
    }

    // Call the ACE-Step generation endpoint through nginx
    fetch('https://doseedo.com/api/generate-ace-step', {
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
            // Create new trackselect for this generation
            const newContainerId = createNewTrackselect('music');

            pollTaskStatus(data.task_id, (result) => {
                createDownloadLinks(result, newContainerId || 'download-links');
            });
            updateStatus('Music Successfully Generated');
        } else {
            console.log('No task ID received');
            updateStatus('Failed to start music generation task.');
        }
    })
    .catch(error => {
        console.error('There was a problem with the fetch operation:', error);
        updateStatus('Error during music generation.');
    });
}

function pollTaskStatus(taskId, callback) {
    fetch(`https://doseedo.com/api/generate-ace-step/task/${taskId}`)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'completed') {
                callback(data.result);
                updateStatus('Task completed. Download available.');
            } else {
                updateStatus(`Processing... (${data.status})`);
                setTimeout(() => pollTaskStatus(taskId, callback), 2000);
            }
        })
        .catch(error => {
            console.error('Error polling task status:', error);
            updateStatus('Error during task status polling.');
        });
}

function createDownloadLinks(filePaths, containerId = 'download-links') {
    let downloadList = document.getElementById(containerId);
    if (!downloadList) {
        console.error('Download list container not found:', containerId);
        return;
    }

    downloadList.innerHTML = "";

    filePaths.forEach((filePath, index) => {
        let rand = Math.floor(Math.random() * (50 - -75 + 1)) + -75;
        const audioLink = `https://doseedo.com/api/generate-ace-step${filePath}`;

        const linkElement = document.createElement("a");
        linkElement.href = audioLink;
        linkElement.style.color = 'white';
        linkElement.innerText = `Audio #${index+1} `;
        linkElement.target = '_blank';
        linkElement.style.float = 'left';

        const listItem = document.createElement("li");
        listItem.appendChild(linkElement);

        const playPauseButton = document.createElement("button");
        playPauseButton.innerText = "▶ ⏸";
        playPauseButton.onclick = () => {
            const video = document.getElementById('player');

            if (window.videoAudioSyncLock) return;
            window.videoAudioSyncLock = true;

            // Play/pause both video and this audio track
            if (wavesurfer.isPlaying()) {
                wavesurfer.pause();
                if (video) video.pause();
            } else {
                wavesurfer.play();
                if (video) video.play();
            }

            window.videoAudioSyncLock = false;
        };
        listItem.appendChild(playPauseButton);

        const downloadimg = document.createElement("i");
        downloadimg.className = 'fa-solid fa-download';
        linkElement.appendChild(downloadimg);

        const waveformContainer = document.createElement("div");
        waveformContainer.className = 'waveform-container';
        listItem.appendChild(waveformContainer);

        const wavesurfer = WaveSurfer.create({
            container: waveformContainer,
            waveColor: `rgb(${100 - rand || 190 + rand}, ${121 + rand || 121 - rand}, ${265 + rand || 265 - rand})`,
            progressColor: 'white',
            backend: 'MediaElement',
            height: 50,
        });

        // Store wavesurfer instance on container for video sync
        waveformContainer.wavesurfer = wavesurfer;

        // Sync audio → video: When audio plays/pauses, sync video
        wavesurfer.on('play', function() {
            const video = document.getElementById('player');
            if (!video) {
                console.warn('⚠️ No video element found');
                return;
            }

            // Skip if we're in a sync operation
            if (window.videoAudioSyncLock) {
                console.log('🔒 Sync locked, skipping');
                return;
            }

            console.log('🎵 Timeline PLAY triggered → playing video');

            if (video.paused) {
                window.videoAudioSyncLock = true;
                console.log('▶️ Starting video playback...');

                video.play().then(() => {
                    console.log('✅ Video playing successfully');
                    window.videoAudioSyncLock = false;
                }).catch(err => {
                    console.error('❌ Video play failed:', err);
                    window.videoAudioSyncLock = false;
                });
            }
        });

        wavesurfer.on('pause', function() {
            const video = document.getElementById('player');
            if (!video) return;

            if (window.videoAudioSyncLock) return;

            console.log('🎵 Timeline PAUSE triggered → pausing video');

            if (!video.paused) {
                window.videoAudioSyncLock = true;
                video.pause();
                console.log('⏸️ Video paused');
                window.videoAudioSyncLock = false;
            }
        });

        // Sync audio seeking with video
        wavesurfer.on('seek', function(progress) {
            const video = document.getElementById('player');
            if (!video || !video.duration || video.duration <= 0) return;

            if (window.videoAudioSyncLock) return;

            console.log('🎵 Timeline SEEK triggered → seeking video to', progress);

            window.videoAudioSyncLock = true;
            video.currentTime = progress * video.duration;
            window.videoAudioSyncLock = false;
        });

        wavesurfer.load(audioLink);
        downloadList.appendChild(listItem);

        var listItems = downloadList.getElementsByTagName('li');
        for (var i = 0; i < listItems.length; i++) {
            rand = Math.floor(Math.random() * (50 - -75 + 1)) + -75;
            listItems[i].style.backgroundColor = `rgb(${140 + rand}, ${101 + rand}, ${265 + rand}, 0.3)`;
        }
    });
}

function showcues(){
    const buttons = document.querySelector('.buttons');
    const cues = document.querySelector('.cues');
    if (buttons) buttons.style.display = 'none';
    if (cues) cues.style.display = 'block';
}

function goback(){
    const cues = document.querySelector('.cues');
    const buttons = document.querySelector('.buttons');
    if (cues) cues.style.display = 'none';
    if (buttons) buttons.style.display = 'block';
}

// Video file handling - matches doseedo1.html behavior
const videoInput = document.getElementById('videoFile');
const videoPreview = document.getElementById('player');

if (videoInput && videoPreview) {
    videoInput.addEventListener('change', function(event) {
        if (event.target.files.length > 0) {
            const file = event.target.files[0];
            const videoElement = document.getElementById('player');
            const videoContainer = document.querySelector('.video-container');
            const glowContainer = document.querySelector('.glow-container');

            const videoUrl = URL.createObjectURL(file);
            videoElement.src = videoUrl;
            videoElement.style.display = 'block';
            videoElement.load();

            // Hide glow-container and show video-container
            if (glowContainer) glowContainer.style.display = 'none';
            if (videoContainer) videoContainer.style.display = 'block';

            // Update resizer position when video loads (handle only, not timeline)
            videoElement.addEventListener('loadedmetadata', function() {
                if (typeof updateResizerHandleOnly === 'function') {
                    setTimeout(updateResizerHandleOnly, 100);
                }
            }, { once: true });

            uploadVideo();
        }
    });
}

// User management functions
function registerUser() {
    const usernameEl = document.getElementById('register-username');
    const emailEl = document.getElementById('register-email');
    const passwordEl = document.getElementById('register-password');

    if (!usernameEl || !emailEl || !passwordEl) return;

    const username = usernameEl.value;
    const email = emailEl.value;
    const password = passwordEl.value;

    const formData = new FormData();
    formData.append('username', username);
    formData.append('email', email);
    formData.append('password', password);

    fetch('https://doseedo.com/register/', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log('Registration success:', data);
        isAuthenticated = true;
        using = username;
        const subscriptionStatus = data.subscription ? "Pro+" : "Free";
        const statusEl = document.getElementById('user-subscription-status');
        if (statusEl) statusEl.textContent = subscriptionStatus;
        isPro = data.subscription ? true : false;

        localStorage.setItem('isAuth', true);
        localStorage.setItem('username', username);
        localStorage.setItem('ispro', subscriptionStatus);
        window.location.reload();
    })
    .catch(error => {
        console.error('Registration error:', error);
    });
}

function loginUser() {
    const usernameEl = document.getElementById('login-username');
    const passwordEl = document.getElementById('login-password');

    if (!usernameEl || !passwordEl) return;

    const username = usernameEl.value;
    const password = passwordEl.value;

    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    fetch('https://doseedo.com/token/', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log('Login success:', data);
        isAuthenticated = true;
        toggleSignOutButton();
        using = username;

        const userUsernameEl = document.getElementById('user-username');
        if (userUsernameEl) userUsernameEl.textContent = username;

        const subscriptionStatus = data.subscription ? "Pro+" : "Free";
        const statusEl = document.getElementById('user-subscription-status');
        if (statusEl) statusEl.textContent = subscriptionStatus;
        isPro = data.subscription ? true : false;

        localStorage.setItem('isAuth', true);
        localStorage.setItem('username', username);
        localStorage.setItem('ispro', subscriptionStatus);

        const userInfo = document.getElementById('user-info');
        const registerForm = document.getElementById('register-form');
        const loginForm = document.getElementById('login-form');
        const signupButton = document.getElementById('signupbutton');
        const loginButton = document.getElementById('loginbutton');

        if (userInfo) userInfo.style.display = 'block';
        if (registerForm) registerForm.style.display = 'none';
        if (loginForm) loginForm.style.display = 'none';
        if (signupButton) signupButton.style.display = 'none';
        if (loginButton) loginButton.style.display = 'none';
        location.reload();
    })
    .catch(error => {
        console.error('Login error:', error);
    });
}

function signOut() {
    isAuthenticated = false;
    toggleSignOutButton();
    localStorage.removeItem('isAuth');
    localStorage.removeItem('username');
    localStorage.removeItem('ispro');
    window.location.reload();
}

function toggleSignOutButton() {
    const signOutButton = document.getElementById("sign-out-button");
    const userInfo = document.getElementById('user-info');
    const signupButton = document.getElementById('signupbutton');
    const loginButton = document.getElementById('loginbutton');

    if (isAuthenticated) {
        if (signOutButton) signOutButton.style.display = "inline";
    } else {
        if (signOutButton) signOutButton.style.display = "none";
        if (userInfo) userInfo.style.display = 'none';
        if (signupButton) signupButton.style.display = 'inline';
        if (loginButton) loginButton.style.display = 'inline';
    }
}

// Video/Timeline synchronization
(function setupVideoTimelineSync() {
    const video = document.getElementById('player');
    const downloadsDiv = document.querySelector('.downloads');

    if (!video || !downloadsDiv) {
        console.warn('⚠️ Video or downloads element not found - sync not initialized');
        return;
    }

    let isSyncing = false; // Prevent circular updates during timeupdate

    // Track all WaveSurfer instances
    window.getAllWavesurfers = function() {
        const containers = downloadsDiv.querySelectorAll('.waveform-container');
        const instances = [];
        containers.forEach(container => {
            if (container.wavesurfer) {
                instances.push(container.wavesurfer);
            }
        });
        return instances;
    };

    // Smooth timeline cursor update using requestAnimationFrame
    function updateTimelineCursor() {
        const timelineCursor = document.getElementById('timeline-cursor');
        const timelineCursor2 = document.getElementById('timeline-cursor2');

        if (timelineCursor && video.duration > 0) {
            const pixelsPerSecond = getPixelsPerSecond();
            const cursorX = video.currentTime * pixelsPerSecond;

            timelineCursor.style.left = `${cursorX}px`;
            if (timelineCursor2) {
                timelineCursor2.style.left = `${cursorX}px`;
            }
        }

        // Continue animating while video is playing
        if (!video.paused) {
            requestAnimationFrame(updateTimelineCursor);
        }
    }

    // Start smooth cursor animation when video plays
    video.addEventListener('play', function() {
        console.log('🎬 Video playing — starting smooth cursor animation');
        requestAnimationFrame(updateTimelineCursor);
    });

    // Sync video → audio: Update waveforms when video time changes
    video.addEventListener('timeupdate', function() {
        if (isSyncing) return;
        isSyncing = true;

        const waveformContainers = downloadsDiv.querySelectorAll('.waveform-container');
        const videoTime = video.currentTime;
        const videoDuration = video.duration;

        if (videoDuration > 0) {
            const seekPosition = videoTime / videoDuration;

            waveformContainers.forEach((container) => {
                if (container.wavesurfer && !container.wavesurfer.isPlaying()) {
                    // Only seek if not already playing (to avoid fighting)
                    container.wavesurfer.seekTo(seekPosition);
                }
            });
        }

        isSyncing = false;
    });

    // Sync video play/pause state with first audio track
    video.addEventListener('play', function() {
        if (window.videoAudioSyncLock) {
            console.log('🔒 Video play - sync locked');
            return;
        }

        console.log('🎬 Video PLAY triggered → playing timeline');
        window.videoAudioSyncLock = true;

        const wavesurfers = getAllWavesurfers();
        if (wavesurfers.length > 0 && !wavesurfers[0].isPlaying()) {
            wavesurfers[0].play();
            console.log('▶️ Timeline playing');
        }

        window.videoAudioSyncLock = false;
    });

    video.addEventListener('pause', function() {
        if (window.videoAudioSyncLock) {
            console.log('🔒 Video pause - sync locked');
            return;
        }

        console.log('🎬 Video PAUSE triggered → pausing timeline');
        window.videoAudioSyncLock = true;

        const wavesurfers = getAllWavesurfers();
        wavesurfers.forEach(ws => {
            if (ws.isPlaying()) {
                ws.pause();
            }
        });
        console.log('⏸️ Timeline paused');

        window.videoAudioSyncLock = false;
    });

    // Seek video when clicking on timeline
    downloadsDiv.addEventListener('click', function(e) {
        const waveformContainer = e.target.closest('.waveform-container');

        if (waveformContainer && video.duration > 0) {
            const rect = waveformContainer.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const containerWidth = rect.width;
            const seekRatio = clickX / containerWidth;
            const seekTime = seekRatio * video.duration;

            if (window.videoAudioSyncLock) return;
            window.videoAudioSyncLock = true;

            // Seek the video
            video.currentTime = seekTime;

            // Seek all waveforms
            const wavesurfers = getAllWavesurfers();
            wavesurfers.forEach(ws => {
                ws.seekTo(seekRatio);
            });

            window.videoAudioSyncLock = false;
        }
    });

    // Spacebar: Play/Pause both video and audio
    document.addEventListener('keydown', function(e) {
        if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
            e.preventDefault();

            if (window.videoAudioSyncLock) return;
            window.videoAudioSyncLock = true;

            const wavesurfers = getAllWavesurfers();

            if (video.paused) {
                video.play();
                if (wavesurfers.length > 0) {
                    wavesurfers[0].play();
                }
            } else {
                video.pause();
                wavesurfers.forEach(ws => ws.pause());
            }

            window.videoAudioSyncLock = false;
        }
    });

    console.log('✅ Video/Timeline synchronization initialized');
})();
