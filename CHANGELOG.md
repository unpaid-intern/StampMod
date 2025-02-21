# Changelog

## 3.0.2
- maybe the last release of stamps, everything works

## 2.4.0
- Switched to using my new `canvasapi`
- Added canvas rotation and orientation in 3D space (hopefully)
- Added new feature to place at the nearest "best viewing area" when holding `Control` (replaces placing at dock)
- Better preprocessing/processing
- Fixed a save menu memory leak
- Long videos won't load all at once and make your game crash
- Longer multiframe support

## 2.3.2
- Added mean shift mapping as requested from Nova

## 2.3.1
- Improved color accuracy for in-game colors
- Better pattern dither

## 2.3.0
- Added WebM support
- Fixed video preview bugs
- Awesome text write feature by `baltdev`

## 2.2.8
- Video support (`MP4` only currently)
- Uses nearest neighbor for upscaling

## 2.2.7
- Optimizations with `numba` courtesy `baltdev`
- Yet again better preprocessing
- Brightness adjustments now work as intended
- More bugs
- Fixed dock canvas placement bug

## 2.2.6
- Fixed error with vertical right-facing images utilizing 2 canvases *ugh*

## 2.2.5
- Now can utilize 2 canvases per image
- Improved preprocessing
- Changing keybind to `Z` for undo will make it `Ctrl+Z`
- Default keybind will now be `Z`

## 2.2.4
- Fixed a mistake I made...

## 2.2.3
- Took away ability to have images larger than `200x200` as only 2 chalk canvases are allowed server-side now?? *Change I didn’t know about.*

## 2.2.2
- Added disclaimer for `thetamborine` incompatibility

## 2.2.1
- Fix for `k-means` mapping

## 2.2.0
- Placement and menu bug fixes
- Much faster multiframe processing
- *Might be the last update/mod for a while, personal stuff*

## 2.1.0
- Improved preprocessing logic
- Chalks support
- Added grass and canvas to color options
- GIF playback speed options
- Took boost and threshold options away from the user *because the user cannot be trusted to make competent decisions*, done automatically now.
- Fixed for Thunderstore Mod Manager *(again!)* **F*** them**

## 2.0.6
- There was a typo

## 2.0.5
- Compatibility with the **stupid fucking knockoff of r2modman** (*Thunderstore Overwolf Mod Manager that can’t extract WebP images or handle nested folders* **fuck you**)  
- *Please, if you aren’t a robot, use `r2modman` instead. Wtffffffff. Like, it works now, but still...*

## 2.0.4
- Bugfix for off-canvas images over `200x200`

## 2.0.3
- Adding update to hopefully fix extraction issues with `r2modman`, so **download size down from `600MB` to like `200MB` hopefully**
- Removed my awesome machine learning model *(rip)*
- **Also, like, `1.11` hype (it works!)**

## 2.0.2
- Better dock handling system
- Changed default keybind and README
- More menu art
- Fixed antivirus and better launch handling
- Better GIF handling
- Faster launch times
- Completely revised launching system

## 2.0.1
- Using... *gentler* compression??? **Because Thunderstore SUCKS**  
- Made launching it less hard to mess up

## 2.0.0
> **We are getting out of beta with this one!**
- Added **art saving** feature
- Added **keybinds** courtesy of `blueberry wolf`
- Made GIFs **more consistent**
- Discovered that **antivirus SUCKS** *(fuck you McAfee)*
- Improved **preprocessing**
- Added **manual brightness adjustment**
- Made **stamp menu smaller** for the people on CRTs
- Changed location of **saved stamps** to `%appdata%/local/webfishing_stamp_mod` *(or whatever the fuck Linux ppl use, so now everything won’t get deleted every update)*
- **Faster launching executable**
- Added **executable manager**

## 1.0.2
- Got `Ctrl+Z` to function and changed keybinds

## 1.0.1
- General bug fixes

## 1.0.0
- Finally got it to link to **correct executable directory**  
*(after stealing script from `KMod`, shoutout `KMod`)*

## 0.1.2
- **AAAAAAUGH I’m STUPID**

## 0.1.1
- **Still didn’t work!**

## 0.1.0
- **Epic troll!!** *(didn't work)*

---

## Future Updates *(Maybe)*
> *Updates will be slow and may not happen at all, but could include:*
- Separate mod for **art accessibility** *(allows artists to use `MyPaint` to edit in-game canvases or new ones)*
- **Canvas locking** for hosts
- **Better organized** save menu

> *In that order, at which point I think I’m happy to move on to other projects outside of WebFishing. I wanted to make something cool and I did!*
