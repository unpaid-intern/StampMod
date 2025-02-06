# Stamps
A stamp and art saving mod for for WEBFISHING using GDWeave

<p align="center">
  <img src="https://raw.githubusercontent.com/unpaid-intern/StampMod/main/menu_gif.gif" alt="kitty campfire"/>
</p>


# Keybinds
> **Keybinds are fully customizable using the game's built in controls settings, courtesy BlueberryWolfiAPIs**

- **`F4` Key**  
	- Opens my stamp menu executable or brings it to the front.

- **`PLUS` Key (+)**  
	- Spawns a stamp at the cursor position.  
	- Spawns a stamp at the player position if `SHIFT` is held.  
	- Spawns a stamp at the dock if `CTRL` is held.  
	- Spawns a stamp on **Canvas 1**, **Canvas 2**, **Canvas 3**, or **Canvas 4** if combined with `1`, `2`, `3`, or `4`, respectively.

- **`MINUS` Key (-)**  
	- Toggles multiframe playback if a GIF/Video has been placed down.  
	- When `SHIFT` is held, toggles playback speed instead.  
	- When `CTRL` is held, resets GIF/Video to play from frame one.

- **`CTRL + Z` Keys**  
	- Works as **Ctrl+Z**, undoing the last placed stamps in order.
  - If this keybind is changed, CTRL will not need to be held.
  - If this isn't working, try the `BACKSPACE` key.



# General Information
- You can **paste images** into the menu directly or **select a file**
- Supports art saving and image generation if using [chalks](https://thunderstore.io/c/webfishing/p/hostileonion/chalks/) by the wonderful hostileonion
- Stamps are automatically adjusted to be **upright relative to the player camera**
- you might get temporarily kicked for spawning canvases in cove servers (they are considered an illegal actor)
- In-game canvases have a **200x200** resolution. I recommend **keeping stamps below this size**
- **Off-canvas**: a maximum of two chalk canvases (200*200) can be placed
- Saved stamps are located at `/home/user/.local/share/webfishing_stamps_mod` on Linux or `%localappdata%/webfishing_stamps_mod` on Windows
- **DONT** move when placing gifs down
- Be mindful of **performance and lag**, especially if playing GIFs. This is a **server side mod**, and people play on potatoes
- I recommend installing the **Calico mod**!
- Please dont use Thunderstore Mod Manager, I suggest Gale or r2modman

# Image Processing Guide

Here’s a quick overview of how to process your images:

- **Color Match**  
  A simple, reliable option that maps each pixel to the closest chalk color. If you're unsure where to start, pick this for clean, predictable results.

- **K-Means Mapping**  
  Groups similar colors into clusters. Great for noise reduction and simplifying complex images. Adjust the number of clusters to control how stylized the result looks.

- **Dithering**  
  If your image has colors that don’t neatly match the available chalk palette, dithering can help fill those gaps. It creates patterns or “noise” to transition between colors smoothly. There are many dithering methods (Atkinson, Floyd, Jarvis, etc.)—each one handles color transitions differently. Generally:
  - **Hybrid Dither** automatically switches between methods for mixed textures.
  - **Pattern Dither** creates a retro, pixel-art style.
  - **Random Dither** adds noise for a more natural, hand-drawn look.  

Ultimately, you should **experiment** with different methods to find the style that best fits your image!

---

### **Color Options**
Choose what chalk colors to use by **enabling** or **disabling** them.

- Use **RGB** to map an image’s colors to RGB chalk.  
- Use **Blank** to make certain colors fully transparent.
- Use **Placing on Canvas** or **Placing on Grass** to add that color to available colors for increased color accuracy.

### **What is LAB and When to Use It?**

>**LAB** is a color space that better reflects how humans perceive color, improving the accuracy of color matching compared to RGB.

- **Turn LAB On**: For images with subtle gradients or color shifts where precise matching matters.  
- **Turn LAB Off**: For bold, flat colors... I think? If the blues are looking a little too red... 

Basically, if you aren't happy with a result, try turning it off! It's on by default but not always better :3


# Canvas map
If you wish to know the location of a specific canvas, please refer to the below map.
<p align="left">
  <img src="https://github.com/unpaid-intern/StampMod/blob/main/MAP.png?raw=true" alt="Canvas ID Map"/>
</p>


# OH NO ITS NOT LAUNCHING FROM THE GAME (LINUX ONLY)

Follow these steps to get it running:

1. **Open Terminal**  
2. **Navigate** to the directory:
   ```
   /GDWeave/mods/PurplePuppy-Stamps_Linux/imagePawcessor
   ```
3. **Make the file executable** by running:
   ```bash
   chmod +x imagePawcess
   ```
4. **Run the file** with:
   ```bash
   ./imagePawcess
   ```
   **Alternatively, use your File Manager**:
   1. Go to `/GDWeave/mods/PurplePuppy-Stamps_Linux/imagePawcessor`.
   2. Right-click `imagePawcess` and select **Properties**.
   3. Enable **Allow executing as a program**.
   4. Double-click the file to run it.

5. **If issues persist**, please report them on my GitHub, including:
   - Your specific Linux distro  
   - How you're running the game (Wine, Proton, etc.)  


## Installation (for the peeps)
**Ensure to not accidentally download from `Code`**
1. Ensure [GDWeave](https://github.com/NotNite/GDWeave) is installed and working properly.
2. Download [for Windows](https://github.com/unpaid-intern/StampMod/releases/download/PurplePuppy-Stamps/PurplePuppy-Stamps.zip) or [for Linux](https://github.com/unpaid-intern/StampMod/releases/download/PurplePuppy-Stamps/PurplePuppy-Stamps_Linux.zip)
3. Extract to `PurplePuppy-Stamps` (`PurplePuppy-Stamps_Linux` for linux) and be careful to **not rename it**.
4. Download [BlueberryWolfi.API](https://github.com/BlueberryWolf/APIs/releases/latest/download/BlueberryWolfi.APIs.zip) for keybinds and extract to `BlueberryWolfi.APIs`
5. Place folders in `WEBFISHING/GDWeave/Mods/`

## Requires:
- [GDWeave](https://github.com/NotNite/GDWeave/tree/main)
- [BlueberryWolfi.APIs](https://github.com/BlueberryWolf/APIs)
