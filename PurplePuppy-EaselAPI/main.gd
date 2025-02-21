extends Node

onready var _Spawn = preload("res://mods/PurplePuppy-EaselAPI/spawner.gd").new()

var first
var in_game

signal stamp_spawncanvas 
signal stamp_playgif
signal stamp_togglemode
signal stamp_ctrlz
signal stamp_resetgif

func _ready():
	#add_child(_Spawn)
	add_to_group("canvasspawner")
	check_updates()

var chalks = false

func check_updates():
	# Initial check: if a player is present at startup, spawn the spawner.
	var current_scene = get_tree().current_scene

	while true:
		yield(get_tree().create_timer(1), "timeout")
		
		# Update chalks if needed.
		if not chalks:
			chalks = get_node_or_null("/root/hostileonionchalks")
		
		current_scene = get_tree().current_scene
		if current_scene == null:
			print("No current scene found.")
			continue
		
		var _Player = current_scene.get_node_or_null("Viewport/main/entities/player")
		if _Player:
			in_game = true
			# If the player exists but _Spawn is missing, add it.
			if _Spawn == null or not _Spawn.is_inside_tree():
				_Spawn = preload("res://mods/PurplePuppy-EaselAPI/spawner.gd").new()
				add_child(_Spawn)
				print("Player detected. _Spawn node added.")
		else:
			in_game = false
			# If the player no longer exists and _Spawn is present, remove it.
			if _Spawn and _Spawn.is_inside_tree():
				_Spawn.queue_free()
				_Spawn = null
				print("Player removed. _Spawn node removed.")



func spawncanvas(stamppath, framespath):
	if _Spawn:
		emit_signal("stamp_spawncanvas", stamppath, framespath)

func ctrlz():
	if _Spawn:
		emit_signal("stamp_ctrlz")

func togglemode():
	if _Spawn:
		emit_signal("stamp_togglemode")

func playgif():
	if _Spawn:
		emit_signal("stamp_playgif")

func resetgif():
	if _Spawn:
		emit_signal("stamp_resetgif")
