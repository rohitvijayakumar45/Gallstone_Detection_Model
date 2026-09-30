"""
Key Code Tester
Helps identify which key codes work on your system for the review tool
"""
import cv2
import numpy as np

def test_key_codes():
    """Test different key codes to see what works on your system"""

    print("=" * 60)
    print("KEY CODE TESTER")
    print("=" * 60)
    print("\nPress various keys to see their codes.")
    print("Try: Arrow keys, A, D, SPACE, B, G, X, Q, R, S")
    print("Press [ESC] to exit")
    print("=" * 60)

    # Create a simple test window
    img = np.zeros((400, 800, 3), dtype=np.uint8)
    cv2.putText(img, "Press keys to test their codes", (50, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "Press [ESC] to exit", (50, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Key Code Tester', img)

    key_history = []

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC key
            break

        # Get key name
        key_name = "UNKNOWN"
        if key == ord('a'):
            key_name = "A (Previous)"
        elif key == ord('d'):
            key_name = "D (Next)"
        elif key == ord(' '):
            key_name = "SPACE (Next)"
        elif key == ord('b'):
            key_name = "B (Bad)"
        elif key == ord('g'):
            key_name = "G (Good)"
        elif key == ord('x'):
            key_name = "X (Delete)"
        elif key == ord('q'):
            key_name = "Q (Quit)"
        elif key == ord('r'):
            key_name = "R (Reset)"
        elif key == ord('s'):
            key_name = "S (Skip)"
        elif key in [81, 2, 2424832]:
            key_name = "LEFT ARROW (Previous)"
        elif key in [83, 3, 2555904, 2424834]:
            key_name = "RIGHT ARROW (Next)"
        elif key == 127 or key == 8:
            key_name = "DELETE/BACKSPACE (Delete)"

        # Record key
        key_info = f"Key: {key} ({key_name})"
        key_history.append(key_info)

        # Keep only last 10 keys
        if len(key_history) > 10:
            key_history.pop(0)

        # Update display
        img = np.zeros((400, 800, 3), dtype=np.uint8)

        cv2.putText(img, f"Last key pressed: {key_info}", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show recent keys
        y_offset = 100
        for i, info in enumerate(reversed(key_history)):
            cv2.putText(img, info, (50, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 30

        cv2.putText(img, "Press [ESC] to exit", (50, 380),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Key Code Tester', img)

    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("KEY CODE SUMMARY")
    print("=" * 60)
    print("\nTested keys:")
    for info in key_history:
        print(f"  {info}")

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("\nIf arrow keys don't work, use these alternatives:")
    print("  - Use [A] for previous (instead of left arrow)")
    print("  - Use [D] for next (instead of right arrow)")
    print("  - Use [SPACE] for next")
    print("\nThese letter keys should work on all systems!")

if __name__ == "__main__":
    test_key_codes()