import argparse
import sys
from pathlib import Path
from src import database
from src import config
from src import workers

def cmd_run(args):
    """Run media archive synchronization."""
    print("Launching cataloging pipeline...")
    print(f"Input folders: {', '.join(str(p) for p in config.INPUT_FOLDERS)}")
    print(f"Output folder: {config.OUTPUT_FOLDER}")
    print(f"Database file: {config.DB_PATH}")
    
    # Launch pipeline
    workers.run_pipeline(force_reprocess=args.force)

def cmd_list_faces(args):
    """List registered faces."""
    database.init_db()
    faces_list = database.get_all_registered_faces()
    
    if not faces_list:
        print("Face registry database is empty. Run scan to detect faces.")
        return
        
    print(f"{'FACE ID':<15} | {'NAME / IDENTITY':<30} | Embedding (shape)")
    print("-" * 65)
    for face_id, name, emb in faces_list:
        print(f"{face_id:<15} | {name:<30} | {emb.shape}")

def cmd_rename_face(args):
    """Rename face in registry."""
    database.init_db()
    # Check if face_id exists in database
    mapping = database.get_face_name_mapping()
    if args.face_id not in mapping:
        print(f"Error: Face with ID '{args.face_id}' not found in database.")
        sys.exit(1)
        
    old_name = mapping[args.face_id]
    database.update_face_name(args.face_id, args.name)
    print(f"Face {args.face_id} successfully renamed: '{old_name}' -> '{args.name}'")

def cmd_analyze_file(args):
    """Analyze a single file, ignoring DB status, and update records."""
    workers.analyze_single_file(args.file)

def cmd_list_face_groups(args):
    """List unrecognized face clusters grouped by similarity."""
    database.init_db()
    groups = database.get_unrecognized_face_groups()
    if not groups:
        print("No unrecognized face groups found.")
        return

    print(f"\nFound {len(groups)} unrecognized face similarity group(s):")
    print(f"{'GROUP ID':<15} | {'COUNT':<6} | {'AVG CONF':<10} | {'FACE IDS':<30} | {'SOURCE FILES'}")
    print("-" * 90)
    for g in groups:
        face_ids_str = ", ".join(g["face_ids"][:4]) + ("..." if len(g["face_ids"]) > 4 else "")
        source_files_str = ", ".join(Path(s).name for s in g["source_files"][:2]) + ("..." if len(g["source_files"]) > 2 else "")
        print(f"{g['group_id']:<15} | {g['count']:<6} | {g['avg_confidence']:<10} | {face_ids_str:<30} | {source_files_str}")

def cmd_assign_group(args):
    """Assign a group of face IDs to a person name."""
    database.init_db()
    success = database.assign_group_to_person(args.face_ids, args.name)
    if success:
        print(f"Successfully assigned {len(args.face_ids)} face(s) to '{args.name}'.")
    else:
        print(f"Error: Failed to assign face(s) {args.face_ids} to '{args.name}'.")
        sys.exit(1)

def cmd_reset_face(args):
    """Reset a face assignment back to unassigned."""
    database.init_db()
    mapping = database.get_face_name_mapping()
    if args.face_id not in mapping:
        print(f"Error: Face ID '{args.face_id}' not found in database.")
        sys.exit(1)
    database.reset_face_assignment(args.face_id)
    print(f"Face '{args.face_id}' assignment reset to unassigned candidate.")

def cmd_reset_file(args):
    """Reset all face assignments originating from a filename."""
    database.init_db()
    res = database.reset_face_assignments_by_filename(args.file)
    print(f"Reset {res['reset_count']} face assignment(s) for file '{args.file}'.")
    if res["face_ids"]:
        print(f"Reset Face IDs: {', '.join(res['face_ids'])}")

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
        
    parser = argparse.ArgumentParser(
        description="Media Archive Cataloging System with Hybrid Processing and Gemini API"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to execute")
    
    # Subcommand: run
    run_parser = subparsers.add_parser("run", help="Run scanning and cataloging")
    run_parser.add_argument(
        "--force", action="store_true", help="Force re-processing of all files"
    )
    
    # Subcommand: list-faces
    subparsers.add_parser("list-faces", help="List all registered faces in database")
    
    # Subcommand: list-face-groups
    subparsers.add_parser("list-face-groups", help="List unrecognized face clusters grouped by similarity")
    
    # Subcommand: rename-face
    rename_parser = subparsers.add_parser("rename-face", help="Assign human-readable name to Face ID")
    rename_parser.add_argument("face_id", type=str, help="Face identifier (e.g. FACE_ID_01)")
    rename_parser.add_argument("name", type=str, help="New human-readable name (e.g. 'John')")
    
    # Subcommand: assign-group
    assign_group_parser = subparsers.add_parser("assign-group", help="Assign a list of Face IDs to a person name")
    assign_group_parser.add_argument("name", type=str, help="Person name")
    assign_group_parser.add_argument("face_ids", nargs="+", type=str, help="Face identifiers (e.g. FACE_ID_01 FACE_ID_02)")
    
    # Subcommand: reset-face
    reset_face_parser = subparsers.add_parser("reset-face", help="Reset a face assignment back to unassigned")
    reset_face_parser.add_argument("face_id", type=str, help="Face identifier (e.g. FACE_ID_01)")
    
    # Subcommand: reset-file
    reset_file_parser = subparsers.add_parser("reset-file", help="Reset all face assignments for a filename or path")
    reset_file_parser.add_argument("file", type=str, help="Filename or path (e.g. photo1.jpg)")
    
    # Subcommand: analyze-file
    analyze_parser = subparsers.add_parser("analyze-file", help="Analyze a single media file, ignoring DB status")
    analyze_parser.add_argument("file", type=str, help="Filename, subpath, or full path to the media file")
    
    args = parser.parse_args()
    
    try:
        if args.command == "run":
            cmd_run(args)
        elif args.command == "list-faces":
            cmd_list_faces(args)
        elif args.command == "list-face-groups":
            cmd_list_face_groups(args)
        elif args.command == "rename-face":
            cmd_rename_face(args)
        elif args.command == "assign-group":
            cmd_assign_group(args)
        elif args.command == "reset-face":
            cmd_reset_face(args)
        elif args.command == "reset-file":
            cmd_reset_file(args)
        elif args.command == "analyze-file":
            cmd_analyze_file(args)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nExecution error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

