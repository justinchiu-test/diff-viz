const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');

async function captureAnimation() {
    console.log('Starting browser...');
    const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--window-size=1400,1000', '--no-sandbox', '--disable-setuid-sandbox']
    });

    const page = await browser.newPage();
    await page.setViewport({ width: 1400, height: 1000, deviceScaleFactor: 2 });

    // Load the visualization page
    const url = `file://${path.join(__dirname, 'index.html')}`;
    console.log('Loading page:', url);
    await page.goto(url, { waitUntil: 'networkidle0' });

    // Inject a speed multiplier to slow down the animation
    await page.evaluate(() => {
        // Override the delay function to multiply all delays by 3
        const originalDelay = DiffVisualizer.prototype.delay;
        DiffVisualizer.prototype.delay = function(ms) {
            return originalDelay.call(this, ms * 3);
        };
        
        // Also slow down the getTokenDelay function
        const originalGetTokenDelay = DiffVisualizer.prototype.getTokenDelay;
        DiffVisualizer.prototype.getTokenDelay = function(token) {
            return originalGetTokenDelay.call(this, token) * 3;
        };
    });

    // Wait for animation to start
    console.log('Waiting for slowed animation to start...');
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Create frames directory
    if (!fs.existsSync('frames')) {
        fs.mkdirSync('frames');
    }

    console.log('Starting capture of slowed animation...');
    
    let frameCount = 0;
    const captureInterval = setInterval(async () => {
        try {
            // Take screenshot
            const paddedFrame = String(frameCount).padStart(4, '0');
            await page.screenshot({ 
                path: `frames/frame_${paddedFrame}.png`,
                type: 'png'
            });
            frameCount++;
            
            // Check if animation is complete
            const isComplete = await page.evaluate(() => {
                const phaseEl = document.getElementById('currentPhase');
                return phaseEl && phaseEl.textContent.includes('Complete');
            });

            if (isComplete || frameCount > 800) {
                clearInterval(captureInterval);
                console.log(`Captured ${frameCount} frames`);
                await browser.close();
                
                // Create GIF using ffmpeg
                console.log('Creating GIF with natural timing...');
                exec('ffmpeg -framerate 20 -i frames/frame_%04d.png -vf "fps=20,scale=1400:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=single[p];[s1][p]paletteuse=dither=none" -loop 0 diff-animation-natural.gif', 
                    (error, stdout, stderr) => {
                        if (error) {
                            console.error('Error creating GIF:', error);
                            console.log('You can manually create the GIF with:');
                            console.log('ffmpeg -r 20 -i frames/frame_%04d.png -vf "scale=1400:-1" diff-animation.gif');
                        } else {
                            console.log('GIF created successfully!');
                            // Clean up frames
                            exec('rm -rf frames', () => {
                                console.log('Cleaned up frames directory');
                            });
                        }
                    }
                );
            }
        } catch (error) {
            console.error('Error capturing frame:', error);
        }
    }, 50); // Capture every 50ms (20 FPS)
}

// Check if Puppeteer is installed
try {
    require.resolve('puppeteer');
    console.log('Puppeteer found. Starting capture...');
    captureAnimation().catch(console.error);
} catch (e) {
    console.error('Puppeteer not found. Please install it first.');
    process.exit(1);
}