extends Node


onready var _Spawn = preload("res://mods/PurplePuppy-Stamps/puppyspawn.gd").new()


var stupidincompatabilitydontchangeshitthatbreaksotherpeoplesmods = null
var debug = false

var KeybindsAPI = null
var img_path = null
var frames_path = null
var gui_path = null
var data_path = null
var cnfg_path = null
var sent = false

var key_states = {}
var in_game = false
var chalks = null

signal spawn_stamp
signal open_menu
signal _delete
signal _play
signal get_data

var plactor = null

var prefix = "[Stamps Config Handler] "


var config_data = {}

var default_config_data = {
	"open_menu": 16777247, 
	"spawn_stamp": 61, 
	"ctrl_z": 16777220, 
	"toggle_playback": 45, 
	"gif_ready": true, 
	"chalks": false,
	"host": false,
	"locked_canvas": [],
	"walky_talky_webfish": "nothing new!", 
	"walky_talky_menu": "nothing new!"
}

func _ready():
	KeybindsAPI = get_node_or_null("/root/BlueberryWolfiAPIs/KeybindsAPI")
	chalks = get_node_or_null("/root/hostileonionchalks")
	load_or_create_config()
	set_chalks()
	img_path = _get_stamp_location()
	frames_path = _get_frames_location()
	gui_path = _get_gui_location()
	data_path = _get_data_out()
	cnfg_path = _get_config_location()
	print("[Stamps] Main script is loaded successfully!")
	add_to_group("keys")
	add_child(_Spawn)
	check_updates()
	if KeybindsAPI:
		var open_menu_signal = KeybindsAPI.register_keybind({
			"action_name": "open_menu", 
			"title": "Open Stamps Menu", 
			"key": config_data["open_menu"], 
		})
		
		var spawn_stamp_signal = KeybindsAPI.register_keybind({
			"action_name": "spawn_stamp", 
			"title": "Spawn Stamp", 
			"key": config_data["spawn_stamp"], 
		})
		
		var ctrl_z_signal = KeybindsAPI.register_keybind({
			"action_name": "ctrl_z", 
			"title": "Stamp Ctrl Z", 
			"key": config_data["ctrl_z"], 
		})
		
		var toggle_playback_signal = KeybindsAPI.register_keybind({
			"action_name": "toggle_playback", 
			"title": "Toggle Gif Playback", 
			"key": config_data["toggle_playback"], 
		})
		
		
		KeybindsAPI.connect(open_menu_signal, self, "_menu")
		KeybindsAPI.connect(spawn_stamp_signal, self, "_stamp")
		KeybindsAPI.connect(toggle_playback_signal, self, "_gif")
		KeybindsAPI.connect(ctrl_z_signal, self, "_ctrlz")
		KeybindsAPI.connect("_keybind_changed", self, "_on_keybind_changed")
	else:
		print("[Stamps] KeybindsAPI not found! Falling back to direct key listening.")
		reset_config()
		
func _process(delta):
	if KeybindsAPI:
		pass
	else:
		check_key_presses()
			

func set_chalks():
	# Check if the 'chalks' node exists and set the config accordingly
	if !chalks:
		chalks = false
	else:
		chalks = true
		
	config_data["chalks"] = chalks
	save_config()
	
	print(prefix, "Chalks set to: ", config_data["chalks"])

func check_key_presses():
	if Input.is_key_pressed(config_data["open_menu"]):
		handle_menu()
	else:
		key_states["open_menu"] = false
		
	if Input.is_key_pressed(config_data["spawn_stamp"]):
		handle_stamp()
	else:
		key_states["spawn_stamp"] = false
		
	if Input.is_key_pressed(config_data["toggle_playback"]):
		handle_gif()
	else:
		key_states["toggle_playback"] = false
		
	if Input.is_key_pressed(config_data["ctrl_z"]):
		handle_ctrlz()
	else:
		key_states["ctrl_z"] = false


func handle_menu():
	if not key_states.has("open_menu"):
		key_states["open_menu"] = false

	if not key_states["open_menu"]:
		key_states["open_menu"] = true
		_menu()


func handle_stamp():
	if not key_states.has("spawn_stamp"):
		key_states["spawn_stamp"] = false

	if not key_states["spawn_stamp"]:
		key_states["spawn_stamp"] = true
		_stamp()
		

func handle_gif():
	if not key_states.has("toggle_playback"):
		key_states["toggle_playback"] = false

	if not key_states["toggle_playback"]:
		key_states["toggle_playback"] = true
		_gif()


func handle_ctrlz():
	if not key_states.has("ctrl_z"):
		key_states["ctrl_z"] = false

	if not key_states["ctrl_z"]:
		key_states["ctrl_z"] = true
		_ctrlz()


func _get_gdweave_dir()->String:
	if debug:
		return "D:/Trash/GDWeave"
	else:
		var game_directory: = OS.get_executable_path().get_base_dir()
		var folder_override: String
		var final_directory: String
		for argument in OS.get_cmdline_args():
			if argument.begins_with("--gdweave-folder-override="):
				folder_override = argument.trim_prefix("--gdweave-folder-override=").replace("\\", "/")
		if folder_override:
			var relative_path: = game_directory.plus_file(folder_override)
			var is_relative: = not ":" in relative_path and Directory.new().file_exists(relative_path)
			final_directory = relative_path if is_relative else folder_override
		else:
			final_directory = game_directory.plus_file("GDWeave")
		return final_directory
		
func _get_stamp_location()->String:
	var gdweave_dir = _get_gdweave_dir()
	var config_path = gdweave_dir.plus_file("mods").plus_file("PurplePuppy-Stamps").plus_file("game_data").plus_file("current_stamp_data").plus_file("stamp.txt")
	return config_path
	
func _get_frames_location()->String:
	var gdweave_dir = _get_gdweave_dir()
	var config_path = gdweave_dir.plus_file("mods").plus_file("PurplePuppy-Stamps").plus_file("game_data").plus_file("current_stamp_data").plus_file("frames.txt")
	return config_path

func _get_gui_location()->String:
	var gdweave_dir = _get_gdweave_dir()
	var executable_name = "imagePawcessor"
	
	
	if OS.get_name() == "Windows":
		executable_name += ".exe"
	
	
	var config_path = gdweave_dir.plus_file("mods").plus_file("PurplePuppy-Stamps").plus_file("imagePawcessor").plus_file(executable_name)
	return config_path

	
func _get_config_location()->String:
	var gdweave_dir = _get_gdweave_dir()
	var config_path = gdweave_dir.plus_file("configs").plus_file("PurplePuppy.Stamps.json")
	return config_path
	
func _get_config_dir()->String:
	var gdweave_dir = _get_gdweave_dir()
	var config_path = gdweave_dir.plus_file("configs")
	return config_path


func _get_data_out()->String:
	var gdweave_dir = _get_gdweave_dir()
	var config_path = gdweave_dir.plus_file("mods").plus_file("PurplePuppy-Stamps").plus_file("game_data").plus_file("game_canvises").plus_file("game_canvises.json")
	return config_path



func load_or_create_config():
	var config_path = _get_config_location()
	var dir = Directory.new()
	
	
	var config_dir = _get_config_dir()
	if not dir.dir_exists(config_dir):
		if dir.make_dir_recursive(config_dir) == OK:
			print(prefix, "Created config directory at: ", config_dir)
		else:
			print(prefix, "Failed to create config directory at: ", config_dir)
			return 
	
	var file = File.new()
	
	if file.file_exists(config_path):
		
		if file.open(config_path, File.READ) == OK:
			var data = file.get_as_text()
			file.close()
			
			
			var json_result = JSON.parse(data)
			if json_result.error == OK and typeof(json_result.result) == TYPE_DICTIONARY:
				config_data = json_result.result
				print(prefix, "Config loaded successfully: ", config_data)
				if config_data.size() != 10:
					print(prefix, "Invalid Config Size, Resetting")
					reset_config()
				return 
			else:
				print(prefix, "Failed to parse config file. Using default config.")
				config_data = default_config_data.duplicate()
		else:
			print(prefix, "Failed to open config file for reading. Using default config.")
			config_data = default_config_data.duplicate()
	else:
		
		config_data = default_config_data.duplicate()
		print(prefix, "Config file created with default values at: ", config_path)
	
	if not config_data.has("walky_talky_menu") or str(config_data["walky_talky_menu"]) != "nothing new!":
		config_data["walky_talky_menu"] = "nothing new!"
	
	save_config()

func reset_config():
	var config_path = _get_config_location()
	var file = File.new()
	
	
	config_data = default_config_data.duplicate()
	
	
	if file.open(config_path, File.WRITE) == OK:
		var json_string = JSON.print(config_data, "	")
		file.store_string(json_string)
		file.close()
		print(prefix, "Configuration reset to default values.")
	else:
		print(prefix, "Failed to open config file for writing during reset.")
		


func save_config():
	var config_path = _get_config_location()
	var file = File.new()

	
	if file.open(config_path, File.WRITE) == OK:
		
		var json_string = JSON.print(config_data, "	")
		file.store_string(json_string)
		file.close()
		print(prefix, "Config saved successfully to: ", config_path)
	else:
		print(prefix, "Failed to open config file for writing: ", config_path)


func get_action_scancode(action_name: String)->int:
	if config_data.has(action_name):
		return config_data[action_name]
	else:
		print(prefix, "Action name not found: ", action_name)
		return - 1
		


func check_updates():
	var was_player_present = false
	var config_path = _get_config_location()
	var file = File.new()

	while true:
		yield (get_tree().create_timer(2.6), "timeout")
		
		if !stupidincompatabilitydontchangeshitthatbreaksotherpeoplesmods:
			stupidincompatabilitydontchangeshitthatbreaksotherpeoplesmods = get_node_or_null("/root/officerballsthetambourine")
			
		if !chalks:
			chalks = get_node_or_null("/root/hostileonionchalks")
			if chalks:
				set_chalks()
		
		var needs_update = false
		if file.open(config_path, File.READ) == OK:
			var data = file.get_as_text()
			file.close()
			
			var json_result = JSON.parse(data)
			if json_result.error == OK and typeof(json_result.result) == TYPE_DICTIONARY:
				config_data = json_result.result

				
				if config_data.has("walky_talky_webfish") and str(config_data["walky_talky_webfish"]) != "nothing new!":
					var webfish_value = str(config_data["walky_talky_webfish"])
					handle_walky_talky_webfish(webfish_value)
					config_data["walky_talky_webfish"] = "nothing new!"
					needs_update = true

				
				if needs_update:
					if file.open(config_path, File.WRITE) == OK:
						var updated_json = JSON.print(config_data, "	")
						file.store_string(updated_json)
						file.close()
						print(prefix, "Config file updated successfully.")
					else:
						print(prefix, "Failed to write updated config.")
			else:
				print(prefix, "Failed to parse config file.")
				load_or_create_config()
		else:
			print(prefix, "Failed to open config file.")
			load_or_create_config()
			
		
		var current_scene = get_tree().current_scene
		if current_scene == null:
			print(prefix, "No current scene found.")
			continue

		var _Player = current_scene.get_node_or_null("Viewport/main/entities/player")
		if _Player == null:
			in_game = false
			if was_player_present:
				if _Spawn:
					_Spawn.queue_free()
				_Spawn = preload("res://mods/PurplePuppy-Stamps/puppyspawn.gd").new()
				add_child(_Spawn)
				print(prefix, "Player was removed. Respawned _Spawn node.")
			was_player_present = false
		else:
			if !was_player_present:
				send_keybind()
				if stupidincompatabilitydontchangeshitthatbreaksotherpeoplesmods:
					#init_player()
					pass
			was_player_present = true
			in_game = true



func handle_walky_talky_webfish(value):
	
	var handlers = {
		"get the canvas data bozo": "handle_get_data"
	}

	if handlers.has(value):
		call(handlers[value])
	else:
		var message = "dude idk what " + str(value) + " means"
		PlayerData._send_notification(message, 1)


func handle_get_data():
	emit_signal("get_data")
	print(prefix, "get_data signal emitted.")



func _on_keybind_changed(action_name: String, title: String, input_event: InputEvent)->void :
	if action_name == "":
		return 
	
	
	if input_event is InputEventKey:
		var scancode = input_event.scancode
		print(prefix, "Action Name:", action_name, "Key Scancode:", scancode)
		
		
		if config_data.has(action_name):
			config_data[action_name] = scancode
			save_config()
		else:
			print(prefix, "Action name not found in config: ", action_name)
	else:
		print(prefix, "Input event is not a key event.")

		
func _menu():
	emit_signal("open_menu")
	
func _stamp():
	emit_signal("spawn_stamp")
	
func _gif():
	emit_signal("_play")

func _ctrlz():
	emit_signal("_delete")
	

func init_player(player: Actor):
	var loadedin = false
	if not loadedin: for i in 5:
		if loadedin: break
		yield (get_tree().create_timer(1), "timeout")
		if get_tree().get_nodes_in_group("controlled_player").size() > 0:
			for actor in get_tree().get_nodes_in_group("controlled_player"):
				if not is_instance_valid(actor): return 
				else:
					if not loadedin:
						plactor = actor
						loadedin = true
	
func send_keybind():
	var key_name = OS.get_scancode_string(config_data["open_menu"])
	if key_name == "":
		return
	yield (get_tree().create_timer(5), "timeout")
	PlayerData._send_notification("Press " + key_name + " for Stamps Menu :3", 0)
