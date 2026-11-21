# Aurora Viewer (Otoa's Mod)

This is not the original Aurora Viewer. The original project is at https://github.com/Aurora-Viewer/Aurora-Viewer

This is my personal clone of it with my own patches, shared only so my patched source can be inspected. It is not for others to use: no builds, no support, and it is not part of the Aurora Viewer project.

## Version

This clone is caught up with the original at `v0.9.13-beta.20261121.1+22` (commit `a1be03f85`).

## Licences

Aurora Viewer is under the GNU General Public License, version 3 or later; TRust, the web engine that patch 77 can use, is under the Mozilla Public License 2.0. My patches are under Aurora's licence, so you are welcome to take any of them if you find them useful.

## Patches used

Each one is a file in scripts/patches/, applied at build time.

- 10-telemetry: no update checks, login news, crash or bug reports, or GIF service counters. The statistics report to Linden Lab stays on, as its viewer policy requires.
- 20-spacemouse: SpaceMouse works on Linux without a udev rule.
- 25-flycam-keys: the keyboard still moves the avatar while the joystick flycam is on.
- 27-script-dialog: a setting for where script dialogs appear.
- 28-credentials: the saved login is sealed with a local tool instead of the desktop keyring.
- 30-audit: small security fixes from an audit (grid links, media filter, link previews, plugin memory).
- 35-sl-grids-only: logs in to the two Second Life grids only.
- 36-channel-name: logs in as "Aurora Viewer (Otoa's Mod)", never as the original viewer.
- 40-clippy: the place for fixes found by the Clippy lint check. It is empty, because the check found nothing to fix.
- 45-moap-slurl: secondlife:// links clicked on in-world web pages follow the Legacy Viewer's safety rules for untrusted pages.
- 50-aurora-shader: a host for user post-processing effects; effects can also read the depth of transparent faces (contract v2).
- 51-aurora-shader-prefs: a Preferences page and a toolbar button for those effects; the page has one tab per effect, dragged to reorder.
- 52-shader-photo-look: those effects as one look in the photo look stack.
- 55-bandwidth-cap: the bandwidth setting goes up to 30,000 kb/s.
- 65-chat-import: the chat import window follows the chat logging settings.
- 76-cef-site-profiles: Chromium keeps cookies only for listed sites.
- 77-trust-engine: TRust as a second web engine beside Chromium. Its host program is not in this repository.
- 78-url-log: an optional local log of the web addresses of media and chat, for a separate checking program; off by default.
- 80-window-shadow: a window shadow strength slider.
- 82-media-off: with media turned off, nothing about media leaves the viewer.
- 83-channel-switches: a sound channel can be turned off completely, not only muted.
- 84-media-filter-switches: two more media filter switches, "Don't report blocked" and "Don't ask".
- 85-media-filter-guard: a check that the media filter stays on for all media; it changes nothing today.
- 86-minimize-stall: a minimized window is no longer reported as a stalled graphics card.
- 87-region-crossing-origin: no stall at region crossings.
- 88-lockfile-advisories: a dependency moved past a security advisory.
- 89-draw-distance-1024: the photo boost raises the draw distance to 1,024 m (upstream now has the 1,024 m range itself).
- 90-scene-impostors: optional far-field impostors, off by default.
- 91-caustics-water-texture: optional underwater caustics from the water's own texture, off by default.
- 92-anim-restart-legacy: an animation signalled again keeps playing instead of restarting.
- 93-ao-over-scripted-overrides: the animation overrider wins over scripted overrides.
- 94-ao-sit-cover-order: the overrider's sit animation shows over its cover pose.
- 95-ao-repair-any-loop: the overrider also repairs lost one-frame poses.
- 96-follow-sim-list: with "Synchronize" on, animations follow the simulator's list exactly.
- 97-new-object-textures: default texture, normal map and specular map for new objects.
- 98-animation-explorer-played: an accurate "Played" column in the Animation explorer.
- 99-hover-leave: the world hover stops when the cursor leaves the window.
- 100-feet-height: a "Feet height" slider from -1 to +1 m, local only.
- 101-chat-scroll-speed: chat lines slide up instead of jumping.
- 102-year-first-dates: dates read year, month, day.
- 103-inventory-display: two inventory display switches: hide empty system folders, and bold folders that hold worn items.
- 104-ao-list-height: the animation list of the AO window grows with the window.
- 105-weather-size: snowflakes can be up to three times their usual size, and a new "Drop size" slider sizes the raindrops the same way.
- 106-director-readd: in the Director, a keyframe added right after deleting the first or the last one goes back where the deleted one was.
- 109-ocean-swell: optional long rolling swells on open water, away from land, their troughs down to a chosen depth, with floating objects and swimmers optionally riding them on this screen only; off by default.
- 111-swim-feet: Aurora foot placement rests while your own avatar swims and comes back by itself when the swim ends.
- 112-voice-neighbours: nearby voice is heard across region borders: on the region-wide channel, a connection to each neighbouring region within reach, the microphone sent only to your own region's.
- 113-director-live-frames: a Director Live film fetches several frames from the graphics card at once (a "Frames fetched at once" setting, 1 to 10, default 3; 1 is the original one at a time) and ignores the frame-rate limits while it records, for a smoother film; frames still on their way when it stops are kept.
- 114-sky-rays: optional volumetric light rays from the sun or the moon, drawn through the shadows; off by default. The code is ported from the commit https://github.com/rickrizzlers/Aurora-Sun-Moon-Preview/commit/c5d71a4a61c9ef7e6f280dc2059ca87703e755b1
- 115-grid-features-opt-in: two upstream features that talk to the grid on their own are off by default: the direct-message history fetched from the grid at login, and Game Controls input sent to the region; both stay switchable in Preferences where upstream puts them.
- 116-hide-owner-objects: "Hide all objects of <owner>" in the object menu hides every rezzed object of that resident (or of the group of a deeded object), listed and undone in the Asset blacklist window; the owner is asked of the region only while such a rule exists, a few at a time. Also: blocking an object stops its particles, and blocking the owner of a group-owned object whose group name is unknown says so instead of doing nothing.
- 117-water-height-offsets: "Flotation height" of swimming from -1 to +1 m, a "Surface swimming height" under it (-1 to +1 m added to the flotation height while swimming along the surface, eased in and out, so the floating and swimming animations each sit right), and a "Water walking height" beside them (-1 to +1 m, below 0 wading); all are real positions the region and others see, sent in the existing bridge command (the worn script is unchanged).
- 118-radio-strict-certificates: an `https` parcel radio whose certificate fails verification (expired, self-signed, untrusted issuer, another name) is refused again instead of played; the radio button's red sign, its tooltip and one nearby chat line per station say why. No setting; a guard fails the patch run if a lenient verifier comes back in the audio code.
- 119-mouselook-smoothing: a "Mouselook smoothing" choice in Preferences (Off by default; Frame average, Exponential, One Euro, Damped spring, Leading cursor) with one Strength slider; the avatar turns with the smoothed view, so others see what you see, and nothing sent changes rate; in Leading cursor mode the crosshair leads and clicks go where it is.
