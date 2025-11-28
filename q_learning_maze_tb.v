`timescale 1ns/1ps

module q_learning_maze_tb;

    // Clock and reset
    reg clk;
    reg reset;

    // DUT signals
    reg  [5:0] s_in;
    reg  [1:0] action_in;
    reg  [5:0] s_next_in;
    reg  signed [15:0] reward_in;
    reg  update_en;
    wire signed [15:0] new_Q_value;
    wire new_Q_valid;

    // Instantiate Q-learning module
    q_update_single_agent #(
        .ALPHA(16'h001A),   // 0.1 in Q8.8
        .GAMMA(16'h00F3)    // 0.95 in Q8.8
    ) dut (
        .clk(clk),
        .reset(reset),
        .s(s_in),
        .a(action_in),
        .s_next(s_next_in),
        .reward(reward_in),
        .update_en(update_en),
        .new_Q_value(new_Q_value),
        .new_Q_valid(new_Q_valid)
    );

    // Clock generation
    always #5 clk = ~clk;

    // Training parameters
    parameter EPISODES = 5000;
    parameter MAX_STEPS = 100;

    // Epsilon (Q8.8 format) - starts at 1.0
    reg [15:0] epsilon;

    // Random seed
    integer seed = 42;

    // Loop variables
    integer ep, step, i, j;

    // Agent state
    reg [5:0] current_state;
    reg [1:0] chosen_action;
    reg [5:0] next_state;
    reg signed [15:0] reward;
    reg done;

    // Maze definition (8x8 grid, 1=wall, 0=path)
    reg [0:63] MAZE;

    // Statistics
    integer total_steps;
    integer success_count;

    initial begin
        $dumpfile("maze_solution.vcd");
        $dumpvars(0, q_learning_maze_tb);
    end

    initial begin
        // Initialize
        clk = 0;
        reset = 1;
        update_en = 0;
        epsilon = 16'd256; // 1.0 in Q8.8

        // Define maze (same as your original)
        // Row 0: 0 0 0 0 0 1 0 0
        // Row 1: 0 1 1 1 0 1 0 0
        // Row 2: 0 1 0 0 0 1 0 0
        // Row 3: 0 1 0 1 1 1 1 0
        // Row 4: 0 0 0 1 0 0 0 0
        // Row 5: 0 1 1 1 0 1 1 0
        // Row 6: 0 0 0 0 0 1 0 0
        // Row 7: 1 0 0 1 0 0 0 0
        
        MAZE = 64'b0;
        // Row 0
        MAZE[5] = 1;
        // Row 1
        MAZE[9] = 1; MAZE[10] = 1; MAZE[11] = 1; MAZE[13] = 1;
        // Row 2
        MAZE[17] = 1; MAZE[21] = 1;
        // Row 3
        MAZE[25] = 1; MAZE[27] = 1; MAZE[28] = 1; MAZE[29] = 1; MAZE[30] = 1;
        // Row 4
        MAZE[35] = 1;
        // Row 5
        MAZE[41] = 1; MAZE[42] = 1; MAZE[43] = 1; MAZE[45] = 1; MAZE[46] = 1;
        // Row 6
        MAZE[53] = 1;
        // Row 7
        MAZE[56] = 1; MAZE[59] = 1;

        #20 reset = 0;
        #20;

        $display("╔════════════════════════════════════════╗");
        $display("║   Single-Agent Maze Q-Learning         ║");
        $display("╚════════════════════════════════════════╝");
        $display("");
        $display("Maze Layout (S=Start at 0, G=Goal at 63):");
        display_maze();
        $display("");

        success_count = 0;
        
        // Training loop
        for (ep = 0; ep < EPISODES; ep = ep + 1) begin
            current_state = 6'd0; // Start position
            done = 0;
            total_steps = 0;

            for (step = 0; step < MAX_STEPS && !done; step = step + 1) begin
                // Choose action (epsilon-greedy)
                choose_action(current_state, chosen_action);

                // Take step in environment
                env_step(current_state, chosen_action, next_state, reward, done);

                // Send update to hardware
                s_in = current_state;
                action_in = chosen_action;
                s_next_in = next_state;
                reward_in = reward;
                update_en = 1;

                @(posedge clk);
                #1;
                update_en = 0;

                // Move to next state
                current_state = next_state;
                total_steps = total_steps + 1;

                if (done) begin
                    success_count = success_count + 1;
                end
            end

            // Decay epsilon
            epsilon = (epsilon * 995) / 1000;
            if (epsilon < 16'd13) epsilon = 16'd13; // Min 0.05

            // Progress updates
            if (ep % 500 == 0 || ep == EPISODES-1) begin
                $display("Episode %5d | Steps: %3d | Epsilon: %0d.%02d | Success Rate: %0d%%", 
                    ep, total_steps, epsilon >> 8, (epsilon & 8'hFF) * 100 / 256,
                    (success_count * 100) / (ep + 1));
            end
        end

        $display("");
        $display("╔════════════════════════════════════════╗");
        $display("║         Training Complete!             ║");
        $display("╚════════════════════════════════════════╝");
        $display("Total Success Rate: %0d%%", (success_count * 100) / EPISODES);
        $display("");

        // Validation run (greedy policy)
        $display("═══════════════════════════════════════");
        $display("  GREEDY POLICY VALIDATION");
        $display("═══════════════════════════════════════");
        
        current_state = 0;
        step = 0;

        while (current_state != 63 && step < 50) begin
            choose_best_action(current_state, chosen_action);
            
            $write("Step %2d: State %2d ", step, current_state);
            display_action(chosen_action);
            
            env_step(current_state, chosen_action, next_state, reward, done);
            
            if (next_state == current_state) 
                $write(" → [BLOCKED]");
            else 
                $write(" → State %2d", next_state);
            
            $display("");
            
            current_state = next_state;
            step = step + 1;
        end

        $display("");
        if (current_state == 63) begin
            $display("✓ SUCCESS: Reached goal in %0d steps!", step);
        end else begin
            $display("✗ FAILED: Could not reach goal in 50 steps");
        end

        $display("");
        #50 $finish;
    end

    // ========================================
    // Environment Step (Maze Logic)
    // ========================================
    task env_step;
        input [5:0] state;
        input [1:0] action;
        output [5:0] next_s;
        output signed [15:0] rew;
        output done_flag;
        
        reg [2:0] row, col;
        reg [2:0] next_row, next_col;
        reg [5:0] potential_next;
        
        begin
            row = state[5:3];
            col = state[2:0];
            next_row = row;
            next_col = col;

            // Calculate next position
            case (action)
                2'd0: if (row > 0) next_row = row - 1; // UP
                2'd1: if (row < 7) next_row = row + 1; // DOWN
                2'd2: if (col > 0) next_col = col - 1; // LEFT
                2'd3: if (col < 7) next_col = col + 1; // RIGHT
            endcase

            potential_next = {next_row, next_col};

            // Check validity
            if (potential_next == state) begin
                // Hit boundary
                next_s = state;
                rew = -16'sd512; // -2.0
                done_flag = 0;
            end
            else if (MAZE[potential_next] == 1'b1) begin
                // Hit wall
                next_s = state;
                rew = -16'sd512; // -2.0
                done_flag = 0;
            end
            else begin
                // Valid move
                next_s = potential_next;
                if (next_s == 6'd63) begin
                    rew = 16'sd2560; // +10.0 (GOAL!)
                    done_flag = 1;
                end else begin
                    rew = -16'sd256; // -1.0
                    done_flag = 0;
                end
            end
        end
    endtask

    // ========================================
    // Epsilon-Greedy Action Selection
    // ========================================
    task choose_action;
        input [5:0] state;
        output [1:0] action;
        integer rand_val;
        begin
            rand_val = $dist_uniform(seed, 0, 255);
            if (rand_val < epsilon) begin
                // Explore
                action = $dist_uniform(seed, 0, 3);
            end else begin
                // Exploit
                choose_best_action(state, action);
            end
        end
    endtask

    // ========================================
    // Greedy Action Selection
    // ========================================
    task choose_best_action;
        input [5:0] state;
        output [1:0] action;
        integer k;
        reg signed [15:0] best_val;
        integer best_idx;
        begin
            best_idx = 0;
            best_val = dut.Q[state][0];
            for (k = 1; k < 4; k = k + 1) begin
                if (dut.Q[state][k] > best_val) begin
                    best_val = dut.Q[state][k];
                    best_idx = k;
                end
            end
            action = best_idx[1:0];
        end
    endtask

    // ========================================
    // Display Helpers
    // ========================================
    task display_maze;
        integer r, c, idx;
        begin
            for (r = 0; r < 8; r = r + 1) begin
                $write("  ");
                for (c = 0; c < 8; c = c + 1) begin
                    idx = r * 8 + c;
                    if (idx == 0)
                        $write("S ");
                    else if (idx == 63)
                        $write("G ");
                    else if (MAZE[idx])
                        $write("█ ");
                    else
                        $write("· ");
                end
                $display("");
            end
        end
    endtask

    task display_action;
        input [1:0] act;
        begin
            case(act)
                2'd0: $write("↑ UP   ");
                2'd1: $write("↓ DOWN ");
                2'd2: $write("← LEFT ");
                2'd3: $write("→ RIGHT");
            endcase
        end
    endtask

endmodule
