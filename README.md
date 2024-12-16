# Stamps
A stamp and art saving mod for for WEBFISHING using GDWeave

<p align="left">
  <img src="https://github.com/unpaid-intern/StampMod/blob/dcdaf02acb6fb342d23e4b34116d9ebc933d3c84/README.png?raw=true" alt="UNDERTALE CLICKBAIT"/>
</p>



# General Information
- **`WILL FAIL TO DOWNLOAD PROPERLY`** if using McAfee Scanner, SecureAge, VirIT, Skyhigh (SWG), Sangfor Engine Zero, or Zillya antivirus (no it's **not a virus** lol)
- The stamps menu might take a moment to launch the first time
- Currently only works for **Windows**
- You can **paste images** into the menu directly or **select a file**
- Stamps are automatically adjusted to be **upright relative to the player camera**
- In-game canvases have a **200x200** resolution. I recommend **keeping stamps below this size**
- **Off-canvas**: a maximum of four chalk canvases (200*200) can be placed
- Saved stamps are located at `%appdata%/local/webfishing_stamps_mod`
- Be mindful of **performance and lag**, especially if playing GIFs. This is a **server side mod**, and people play on potatoes



# Keybinds
> **Keybinds are fully customizable using the game's built in controls settings, courtesy BlueberryWolfiAPIs**

- **`F9` Key**  
  - Opens my stamp menu executable.

- **`PLUS` Key (+)**  
  - Spawns a stamp at the cursor position.
  - Spawns a stamp at the player position if `SHIFT` is held.
  - Spawns a stamp at the dock if `CTRL` is held.
  - Spawns a stamp on **Canvas 1**, **Canvas 2**, **Canvas 3**, or **Canvas 4** if combined with `1`, `2`, `3`, or `4`, respectively.

- **`Minus` Key (-)**  
  - Toggles GIF playback if a GIF has been placed down.

- **`BACKSPACE` Key**  
  - Works as **Ctrl+Z**, undoing the last placed stamps in order.




# Image Processing Guide

Here’s a quick guide to the different image processing methods and when to use them:

- **Color Match**: Simple and reliable, maps each pixel to the closest chalk color. Great for clean, predictable results. If you're unsure where to start, pick this.

- **K-Means Mapping**: Groups similar colors into clusters. Perfect for reducing noise and simplifying complex images. Adjust the clusters to control how stylized the result looks.

- **Hybrid Dither**: Dynamically switches between Atkinson and Floyd-Steinberg dithering based on texture. Use this for images with a mix of smooth and detailed areas.

- **Pattern Dither**: Uses a repeating 8x8 Bayer matrix for a retro, pixel-art effect. Best for artsy look or... nostalgia?

- **Atkinson Dither**: Lightweight dithering. Great for small images! It was used on early Mac computers for monochrome displays.

- **Stucki Dither**: Smooths gradients with wide error diffusion. Ideal for larger images where you want less noise and softer transitions.

- **Floyd Dither**: A classic dithering method that balances smooth gradients and simplicity. Best for medium to large images.

- **Jarvis Dither**: Similar to Stucki but diffuses errors over an even larger area. Great for very detailed or gradient-heavy images.

- **Sierra Dither**: A faster, simplified alternative to Jarvis. Good for medium images.

- **Random Dither**: Introduces noise for a chaotic, natural texture. Great for breaking up banding or adding a hand-drawn feel.

If you're looking for consistency, use **Color Match**. For more artistic results, experiment with the dithering methods to find what works best!

---

### **Color Options**

Choose what chalk colors to use by **enabling** or **disabling** them.

- Use **RGB** to map an image’s colors to RGB chalk.  
- Use **Blank** to make certain colors fully transparent.

#### **Boost and Threshold Adjustments**
- **Boost:** Enhances the intensity of specific chalk colors in an image, making muted tones stand out more. Higher values = brighter colors.  
- **Threshold:** Adjusts how closely a pixel's color must match a chalk color to be boosted. Lower values are more precise, while higher values affect broader ranges.

---

### **What is LAB and When to Use It?**

>**LAB** is a color space that better reflects how humans perceive color, improving the accuracy of color matching compared to RGB.

- **Turn LAB On**: For images with subtle gradients or color shifts where precise matching matters.  
- **Turn LAB Off**: For bold, flat colors... I think?

Basically, if you aren't happy with a result, try turning it off! It's on by default but not always better~

Sometimes **disabling green chalk** will make LAB results look better **:3**




# Canvas map
If you wish to know the location of a specific canvas, please refer to the below map.
<p align="left">
  <img src="https://github.com/unpaid-intern/StampMod/blob/main/MAP.png?raw=true" alt="Canvas ID Map"/>
</p>



## Installation (for the peeps)
**Ensure to not accidentally download from `Code`**
1. Ensure [GDWeave](https://github.com/NotNite/GDWeave) is installed and working properly.
2. Download [for Windows](https://github.com/unpaid-intern/StampMod/releases/download/PurplePuppy-Stamps/PurplePuppy-Stamps.zip)
3. Extract to `PurplePuppy-Stamps` and be careful to **not rename it**.
4. Download [BlueberryWolfi.API](https://github.com/BlueberryWolf/APIs/releases/latest/download/BlueberryWolfi.APIs.zip) and extract to `BlueberryWolfi.APIs`
5. Place folders in `WEBFISHING/GDWeave/Mods/`

## Requires:
- [GDWeave](https://github.com/NotNite/GDWeave/tree/main)
- [BlueberryWolfi.APIs](https://github.com/BlueberryWolf/APIs)
